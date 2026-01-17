import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter 
from tqdm import tqdm
from data_loader import get_loaders
from model import CNNtoRNN
from utils import seed_everything, Experiment, evaluate_bleu 
import os

def train():
    config = {
        "experiment_name": "ResNet101_LSTM_v1",
        "seed": 42,
        "learning_rate": 3e-4,
        "batch_size": 32, 
        "embed_size": 256,
        "hidden_size": 256,
        "num_layers": 1,
        "num_epochs": 100,
        "save_model": True,
        "num_workers": 16, 
        "image_size": 224,
        "resize_dim": 232,
        "train_cnn": False,
        "bleu_every_n_epochs": 5,
        "load_model": False, 
        "checkpoint_path": "experiments/2026-01-17_19-04-17_ResNet101_LSTM_v1/weights/checkpoint_epoch_3.pth.tar", 
        "system": {
             "device_fallback_enabled": True
        }
    }
    
    seed_everything(config["seed"])
    
    experiment = Experiment(config["experiment_name"], config)
    
    writer = SummaryWriter(log_dir=experiment.logs_dir)

    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.zeros(1).to(device)
            
            try:
                dummy_lstm = nn.LSTM(10, 10).to(device)
            except Exception:
                print("\n[WARNING] Standard GPU LSTM failed. Disabling CuDNN and retrying on GPU...")
                torch.backends.cudnn.enabled = False
                dummy_lstm = nn.LSTM(10, 10).to(device)
                print("[SUCCESS] LSTM initialized on GPU with CuDNN disabled.")
                
            try:
                dummy_conv = nn.Conv2d(3, 3, kernel_size=3).to(device)
                dummy_input = torch.randn(1, 3, 224, 224).to(device)
                _ = dummy_conv(dummy_input)
            except Exception as e:
                print(f"\n[ERROR] GPU Conv2d failed: {e}")
                raise e 
                
            print("[SUCCESS] GPU checks passed. Using CUDA.\n")
        else:
            device = torch.device("cpu")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] GPU hardware/software incompatibility: {e}")
        print("Switching to CPU mode so you can keep working.\n")
        device = torch.device("cpu")

    transform = transforms.Compose(
        [
            transforms.Resize((config["resize_dim"], config["resize_dim"])),
            transforms.CenterCrop((config["image_size"], config["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    root_folder = "caption_data/Images" 
    annotation_file = "caption_data/captions.txt"
    
    print("Initializing Loaders...")
    train_loader, val_loader, test_loader, vocab = get_loaders(
        root_folder=root_folder,
        annotation_file=annotation_file,
        transform=transform,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        test_size=0.1,
        val_size=0.1
    )
    
    torch.save(vocab, os.path.join(experiment.dir, "vocab.pth"))

    vocab_size = len(vocab)
    model = CNNtoRNN(
        config["embed_size"], 
        config["hidden_size"], 
        vocab_size, 
        config["num_layers"]
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    step = 0
    start_epoch = 0

    if config.get("load_model", False) and config["checkpoint_path"]:
        print(f"=> Loading checkpoint: {config['checkpoint_path']}")
        if os.path.exists(config["checkpoint_path"]):
            checkpoint = torch.load(config["checkpoint_path"], map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = checkpoint["epoch"] + 1
            step = 0 
            print(f"=> Resuming from Epoch {start_epoch}")
        else:
            print(f"=> CAUTION: Checkpoint not found at {config['checkpoint_path']}")
            print("   Starting from scratch...")

    for epoch in range(start_epoch, config["num_epochs"]):
        
        model.train()
        print(f"Epoch [{epoch+1}/{config['num_epochs']}] Training...")
        loop = tqdm(train_loader, leave=True)
        train_loss = 0

        for idx, (imgs, captions) in enumerate(loop):
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs, captions[:-1]) 
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            writer.add_scalar("Training/Batch_Loss", loss.item(), step)
            step += 1

            loop.set_description(f"Epoch [{epoch+1}/{config['num_epochs']}]")
            loop.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader)
        print(f"Average Train Loss: {avg_train_loss:.4f}")
        writer.add_scalar("Training/Epoch_Loss", avg_train_loss, epoch)

        model.eval()
        val_loss = 0
        print(f"Epoch [{epoch+1}/{config['num_epochs']}] Validation...")
        with torch.no_grad():
            for imgs, captions in val_loader:
                imgs = imgs.to(device)
                captions = captions.to(device)

                outputs = model(imgs, captions[:-1])
                loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Average Val Loss: {avg_val_loss:.4f}")
        writer.add_scalar("Validation/Epoch_Loss", avg_val_loss, epoch)

        if "bleu_every_n_epochs" not in config:
            config["bleu_every_n_epochs"] = 5
            
        if (epoch + 1) % config["bleu_every_n_epochs"] == 0:
            print("Running BLEU Evaluation...")
            bleu_score = evaluate_bleu(val_loader, model, device, vocab)
            writer.add_scalar("Validation/BLEU_Score", bleu_score, epoch)

        if config["save_model"]:
             checkpoint = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "vocab": vocab, 
                "config": config 
            }
             save_path = experiment.get_checkpoint_path(epoch)
             torch.save(checkpoint, save_path)
             print(f"Checkpoint saved: {save_path}")

if __name__ == "__main__":
    train()
