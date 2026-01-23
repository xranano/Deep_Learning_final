import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import os
import random
from attention_model import AttentionModel
from data_loader import get_loaders, get_transforms, Vocabulary # Explicit import
from torch.utils.data import DataLoader

def evaluate_final():
    print("Evaluating Final Best Model...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data & Vocab
    # We rebuild vocab from scratch via get_loaders which is safer than loading pickle across scripts
    print("Loading Data and Rebuilding Vocabulary...")
    
    # Use standard transform for validation
    transform = get_transforms(train=False)
    
    # get_loaders builds the vocab internally
    train_loader, val_loader, test_loader, vocab = get_loaders(
        root_folder="caption_data/Images",
        annotation_file="caption_data/captions.txt",
        transform=transform,
        batch_size=32,
        num_workers=2
    )
    
    # 2. Initialize Model (Must match training config)
    # Config from train_best_model: Embed=512, Hidden=512, AttentionDim=512
    print("Initializing Model...")
    model = AttentionModel(
        attention_dim=512,
        embed_dim=512,
        decoder_dim=512,
        vocab_size=len(vocab),
        encoder_dim=2048,
        dropout=0.5,
        train_CNN=False 
    ).to(device)
    
    # 3. Load Weights
    # Path found in experiments/
    checkpoint_path = "experiments/2026-01-22_19-39-17_Final_Best_Model_Attention/weights/best_model.pth.tar"
    print(f"Loading weights from {checkpoint_path}...")
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        print("Weights Loaded Successfully!")
    else:
        print("Checkpoint not found!")
        return

    model.eval()
    
    # 4. Generate Predictions
    print("\n" + "="*60)
    print("       FINAL MODEL EVALUATION (10 Samples)       ")
    print("="*60)
    
    output_dir = "final_evaluation_samples"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    num_samples = 10
    
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    to_pil = transforms.ToPILImage()
    
    from PIL import ImageDraw
    # Try to load a nicer font, fallback to default
    try:
         font = ImageFont.truetype("arial.ttf", 16)
    except:
         font = None

    with torch.no_grad():
        for idx, (img, captions) in enumerate(test_loader):
            if idx >= num_samples:
                break
                
            img_tensor = img.to(device)
            
            # Generate Caption
            # Note: caption_image expects single image (3, 224, 224) usually, but logic depends on implementation
            # Checking attention_model.py: caption_image takes (image, vocab) -> unsqueezes inside
            
            # We iterate loader which gives Batch. Let's take first from batch for simplicity or iterate batch?
            # Actually test_loader gives batches. Let's pick the first image from each of the first 10 batches
            # to get random variance.
            
            single_img = img_tensor[0] # Take first image of batch
            gt_caption_indices = captions[0]
            
            pred_caption = model.caption_image(single_img.unsqueeze(0), vocab, device=device)
            pred_text = "Pred: " + " ".join(pred_caption)
            
            # Get Ground Truth text
            truth_tokens = []
            for i in gt_caption_indices:
                word = vocab.itos[i.item()]
                if word == "<EOS>": break
                if word not in ["<SOS>", "<PAD>"]:
                    truth_tokens.append(word)
            truth_text = "Truth: " + " ".join(truth_tokens)
            
            print(f"\n[Sample {idx+1}]")
            print(truth_text)
            print(pred_text)
            print("-" * 30)
            
            # Save Image
            img_vis = inv_normalize(single_img.cpu())
            pil_img = to_pil(img_vis)
            
            w, h = pil_img.size
            new_h = h + 120
            new_img = Image.new("RGB", (w, new_h), (255, 255, 255))
            new_img.paste(pil_img, (0, 0))
            
            draw = ImageDraw.Draw(new_img)
            
            def draw_multiline(text, y_pos, color):
                words = text.split()
                lines = []
                current_line = []
                for word in words:
                    current_line.append(word)
                    if len(" ".join(current_line)) > 40:
                        lines.append(" ".join(current_line[:-1]))
                        current_line = [word]
                lines.append(" ".join(current_line))
                
                for line in lines:
                    draw.text((10, y_pos), line, fill=color, font=font)
                    y_pos += 20
                return y_pos

            y = h + 10
            y = draw_multiline(truth_text, y, (0, 120, 0)) 
            draw_multiline(pred_text, y + 5, (0, 0, 180)) 
            
            save_path = os.path.join(output_dir, f"final_sample_{idx+1}.jpg")
            new_img.save(save_path)
            print(f"Saved {save_path}")

if __name__ == "__main__":
    evaluate_final()
