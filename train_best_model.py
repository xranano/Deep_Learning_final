from train import train
import torch

def train_best():
    print("Training Best Model Configuration (Attention + Optimized Hyperparams)...")
    
    # Based on Trial 4 (Best BLEU: 0.076)
    config = {
        "experiment_name": "Final_Best_Model_Attention",
        "learning_rate": 0.0003, # 3e-4 worked best
        "batch_size": 32,
        "embed_size": 512,       # Higher embedding dim helped
        "hidden_size": 512,      # Higher hidden dim helped
        "num_layers": 1,
        "dropout": 0.5,
        "rnn_type": "LSTM",
        "optimizer": "Adam",
        "model_type": "attention",
        "attention_dim": 512,    # Matched hidden size
        "num_epochs": 20,       # Give it enough time
        "save_model": True,
        "patience": 5,
        "train_cnn": False,      # Start frozen
        "bleu_every_n_epochs": 1,
        "load_model": False,
        # "checkpoint_path": "experiments\\2026-01-22_19-31-53_Final_Best_Model_Attention\\weights\\checkpoint_epoch_0.pth.tar"
    }
    
    # Run training
    results = train(config)
    
    print("Training Complete!")
    print(f"Best BLEU: {results['best_bleu']}")
    
    # --- EVALUATION ---
    print("\nRunning Evaluation...")
    from attention_model import AttentionModel
    from data_loader import get_transforms
    from PIL import Image, ImageDraw, ImageFont
    import os
    import torchvision.transforms as transforms
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = results['vocab'] # Get vocab directly from training result
    
    # Re-init model for inference
    model = AttentionModel(
        attention_dim=512,
        embed_dim=512,
        decoder_dim=512,
        vocab_size=len(vocab),
        encoder_dim=2048,
        dropout=0.5,
        train_CNN=False 
    ).to(device)
    
    checkpoint_path = os.path.join(results['experiment_dir'], "weights", "best_model.pth.tar")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        print("Model generated and weights loaded!")
    
    model.eval()
    
    output_dir = "final_evaluation_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Get test loader
    transform = get_transforms(train=False)
    _, _, test_loader, _ = get_loaders(
        root_folder="caption_data/Images",
        annotation_file="caption_data/captions.txt",
        transform=transform,
        batch_size=1, # Single image for viz
        num_workers=2,
        shuffle=True
    )
    
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    to_pil = transforms.ToPILImage()
    
    try:
         font = ImageFont.truetype("arial.ttf", 20)
    except:
         font = None

    print("Generating 10 samples...")
    with torch.no_grad():
        for idx, (img, captions) in enumerate(test_loader):
            if idx >= 10: break
            
            img = img.to(device)
            # caption_image expects (3, 224, 224) usually but we adapted handle batch?
            # AttentionModel.caption_image expects (1, 3, 224, 224) or (3, 224, 224)?
            # My implementation takes image, passes to encoder. Encoder expects batch.
            # So (1, 3, 224, 224) is correct.
            
            pred_caption = model.caption_image(img, vocab, device=device)
            pred_text = "Pred: " + " ".join(pred_caption)
            
            # GT
            gt_caption_indices = captions[0]
            truth_tokens = []
            for i in gt_caption_indices:
                word = vocab.itos[i.item()]
                if word == "<EOS>": break
                if word not in ["<SOS>", "<PAD>"]:
                    truth_tokens.append(word)
            truth_text = "Truth: " + " ".join(truth_tokens)
            
            print(f"[Sample {idx+1}] {pred_text}")
            
            # Save
            img_vis = inv_normalize(img.squeeze(0).cpu())
            pil_img = to_pil(img_vis)
            
            w, h = pil_img.size
            new_img = Image.new("RGB", (w, h + 100), (255, 255, 255))
            new_img.paste(pil_img, (0, 0))
            draw = ImageDraw.Draw(new_img)
            
            draw.text((10, h + 10), truth_text, fill="green", font=font)
            draw.text((10, h + 40), pred_text, fill="blue", font=font)
            
            new_img.save(os.path.join(output_dir, f"sample_{idx}.jpg"))

if __name__ == "__main__":
    train_best()
