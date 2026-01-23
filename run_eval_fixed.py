import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet101_Weights
import os
import sys

# Ensure data_loader is available globally as 'data_loader'
import data_loader
sys.modules['data_loader'] = data_loader

from attention_model import AttentionModel
from data_loader import get_loaders, get_transforms
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms

def run_eval():
    print("Running Final Evaluation (Fixed)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Rebuild Vocabulary
    print("Rebuilding Vocabulary...")
    transform = get_transforms(train=False)
    _, _, test_loader, vocab = get_loaders(
        root_folder="caption_data/Images",
        annotation_file="caption_data/captions.txt",
        transform=transform,
        batch_size=1,
        num_workers=2,
        shuffle=True
    )
    
    checkpoint_path = "experiments/2026-01-22_19-39-17_Final_Best_Model_Attention/weights/best_model.pth.tar"
    print(f"Loading from {checkpoint_path}")
    
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            vocab = checkpoint['vocab']
            print(f"Loaded vocab from checkpoint. Size: {len(vocab)}")
            
            # 2. Init Model with CORRECT vocab size
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
            
            model.load_state_dict(checkpoint["state_dict"])
            print("Weights loaded successfully!")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            import traceback
            traceback.print_exc()
            return
    else:
        print("Checkpoint not found.")
        return

    model.eval()
    
    # 4. Generate
    print("Generating Samples...")
    output_dir = "final_evaluation_results_fixed"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    to_pil = transforms.ToPILImage()
    
    try:
         font = ImageFont.truetype("arial.ttf", 20)
    except:
         font = None

    with torch.no_grad():
        for idx, (img, captions) in enumerate(test_loader):
            if idx >= 10: break
            
            img_tensor = img.to(device)
            # (1, 3, 224, 224)
            
            pred_caption = model.caption_image(img_tensor, vocab, device=device)
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
            img_vis = inv_normalize(img_tensor.squeeze(0).cpu())
            pil_img = to_pil(img_vis)
            
            w, h = pil_img.size
            new_img = Image.new("RGB", (w, h + 100), (255, 255, 255))
            new_img.paste(pil_img, (0, 0))
            draw = ImageDraw.Draw(new_img)
            
            draw.text((10, h + 10), truth_text, fill="green", font=font)
            draw.text((10, h + 40), pred_text, fill="blue", font=font)
            
            new_img.save(os.path.join(output_dir, f"sample_{idx}.jpg"))
            
if __name__ == "__main__":
    run_eval()
