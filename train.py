import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter 
from tqdm import tqdm
from data_loader import get_loaders, get_transforms
from model import CNNtoRNN
from attention_model import AttentionModel
from utils import print_examples, evaluate_bleu, Experiment, seed_everything
import os

def train(config=None):
    # Default Config
    default_config = {
        "experiment_name": "ResNet101_LSTM_v1",
        "seed": 42,
        "learning_rate": 3e-4,
        "batch_size": 32, 
        "embed_size": 256,
        "hidden_size": 256,
        "num_layers": 1,
        "num_epochs": 100,
        "save_model": True,
        "num_workers": 6, 
        "image_size": 224,
        "resize_dim": 232,
        "train_cnn": False,
        "bleu_every_n_epochs": 5,
        "load_model": False, 
        "checkpoint_path": "experiments/2026-01-17_19-04-17_ResNet101_LSTM_v1/weights/checkpoint_epoch_3.pth.tar", 
        "optimizer": "Adam",
        "rnn_type": "LSTM",
        "patience": 5,
        "system": {
             "device_fallback_enabled": True
        }
    }
    
    # Merge defaults with provided config
    if config:
        # Shallow merge usually enough for flat configs, but be careful with nested dicts
        for key, value in config.items():
            default_config[key] = value
    
    config = default_config
    
    seed_everything(config["seed"])
    
    experiment = Experiment(config["experiment_name"], config)
    
    writer = SummaryWriter(log_dir=experiment.logs_dir)

    # Model Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Loaders
    print("Loading data...")
    # Use robust transforms
    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)
    
    train_loader, val_loader, test_loader, vocab = get_loaders(
        root_folder="caption_data/Images",
        annotation_file="caption_data/captions.txt",
        train_transform=train_transform,
        val_transform=val_transform,
        transform=None, # Legacy ignored
        batch_size=config["batch_size"],
        num_workers=2
    )
    
    torch.save(vocab, os.path.join(experiment.dir, "vocab.pth"))

    print(f"Initializing Model: {config.get('model_type', 'CNNtoRNN')}")

    model_type = config.get("model_type", "CNNtoRNN")

    if model_type == "attention":
        model = AttentionModel(
            attention_dim=config.get("attention_dim", 256),
            embed_dim=config["embed_size"],
            decoder_dim=config["hidden_size"],
            vocab_size=len(vocab),
            encoder_dim=2048,
            dropout=config.get("dropout", 0.5),
            train_CNN=False 
        ).to(device)
    else:
        model = CNNtoRNN(
            config["embed_size"],
            config["hidden_size"],
            len(vocab),
            config["num_layers"],
            rnn_type=config.get("rnn_type", "LSTM") # Default to LSTM if missing
        ).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
    
    # Optimizer Selection
    if config["optimizer"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"] == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"] == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=config["learning_rate"])
    else:
        print(f"Warning: Unknown optimizer {config['optimizer']}, defaulting to Adam.")
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    step = 0
    start_epoch = 0

    if config.get("load_model", False) and config["checkpoint_path"]:
        # Load checkpoint logic...
         if os.path.exists(config["checkpoint_path"]):
            checkpoint = torch.load(config["checkpoint_path"], map_location=device)
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = checkpoint["epoch"] + 1
            print(f"=> Resuming from Epoch {start_epoch}")

    # Early Stopping Variables
    best_val_loss = float("inf")
    patience_counter = 0
    best_bleu_score = 0.0

    for epoch in range(start_epoch, config["num_epochs"]):
        
        # Training Phase
        model.train()
        print(f"Epoch [{epoch+1}/{config['num_epochs']}] Training...")
        # Use simple prints if running automated to avoid log clutter, or keep tqdm if manual
        if "tqdm_disable" in config and config["tqdm_disable"]:
             loop = train_loader
        else:
             loop = tqdm(train_loader, leave=True)
             
        train_loss = 0

        for idx, (imgs, captions) in enumerate(loop):
            imgs = imgs.to(device)
            captions = captions.to(device)

            if model_type == "attention":
                # Attention Model expects (Batch, Seq_Len)
                captions = captions.permute(1, 0)
                
                # Attention Model Forward
                # Calculate lengths (excluding PAD)
                caption_lengths = (captions != vocab.stoi["<PAD>"]).sum(dim=1)
                
                preds, caps_sorted, decode_lens, alphas, sort_ind = model(imgs, captions, caption_lengths)
                
                # Targets are sorted captions shifted by 1 (remove SOS) since preds start from word 1
                targets = caps_sorted[:, 1:]
                
                # Pack sequences for optimized loss calculation
                from torch.nn.utils.rnn import pack_padded_sequence
                
                preds_packed = pack_padded_sequence(preds, decode_lens, batch_first=True).data
                targets_packed = pack_padded_sequence(targets, decode_lens, batch_first=True).data
                
                loss = criterion(preds_packed, targets_packed)
                
                # Optional: Doubly stochastic attention regularization
                # loss += ((1. - alphas.sum(dim=1)) ** 2).mean() * 1.0
                
            else:
                # Standard CNNtoRNN Forward
                outputs = model(imgs, captions[:, :-1])
                loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions[:, 1:].reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            writer.add_scalar("Training/Batch_Loss", loss.item(), step)
            step += 1
            
            if not ("tqdm_disable" in config and config["tqdm_disable"]):
                 loop.set_description(f"Epoch [{epoch+1}/{config['num_epochs']}]")
                 loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        print(f"Average Train Loss: {avg_train_loss:.4f}")
        writer.add_scalar("Training/Epoch_Loss", avg_train_loss, epoch)

        # Validation Phase
        model.eval()
        val_loss = 0
        print(f"Epoch [{epoch+1}/{config['num_epochs']}] Validation...")
        with torch.no_grad():
            for imgs, captions in val_loader:
                imgs = imgs.to(device)
                captions = captions.to(device)

                if model_type == "attention":
                    # Attention Model expects (Batch, Seq_Len)
                    captions = captions.permute(1, 0)
                    
                    caption_lengths = (captions != vocab.stoi["<PAD>"]).sum(dim=1)
                    
                    preds, caps_sorted, decode_lens, alphas, sort_ind = model(imgs, captions, caption_lengths)
                     
                    targets = caps_sorted[:, 1:]
                    
                    from torch.nn.utils.rnn import pack_padded_sequence
                    
                    preds_packed = pack_padded_sequence(preds, decode_lens, batch_first=True).data
                    targets_packed = pack_padded_sequence(targets, decode_lens, batch_first=True).data
                    
                    loss = criterion(preds_packed, targets_packed)
                else:
                    outputs = model(imgs, captions[:, :-1])
                    loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions[:, 1:].reshape(-1))
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Average Val Loss: {avg_val_loss:.4f}")
        writer.add_scalar("Validation/Epoch_Loss", avg_val_loss, epoch)

        # BLEU Evaluation
        if (epoch + 1) % config["bleu_every_n_epochs"] == 0:
            print("Running BLEU Evaluation...")
            bleu_score = evaluate_bleu(val_loader, model, device, vocab)
            writer.add_scalar("Validation/BLEU_Score", bleu_score, epoch)
            if bleu_score > best_bleu_score:
                best_bleu_score = bleu_score

        # Checkpoints
        # Prepare Checkpoint
        checkpoint = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "vocab": vocab, 
            "config": config 
        }

        # Save Checkpoint if enabled
        if config["save_model"]:
             save_path = experiment.get_checkpoint_path(epoch)
             torch.save(checkpoint, save_path)

        # Early Stopping Logic (Based on Loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save "Best" Model distinct from regular checkpoints
            torch.save(checkpoint, os.path.join(experiment.weights_dir, "best_model.pth.tar"))
            print("New Best Model Saved!")
        else:
            patience_counter += 1
            print(f"Early Stopping Counter: {patience_counter}/{config['patience']}")
            
            if patience_counter >= config["patience"]:
                print("Early Stopping Triggered. Stopping Training.")
                break

    print(f"Training Complete. Best Val Loss: {best_val_loss:.4f}, Best BLEU: {best_bleu_score:.2f}")
    return {
        "best_val_loss": best_val_loss,
        "best_bleu": best_bleu_score,
        "experiment_dir": experiment.dir
    }

if __name__ == "__main__":
    train()
