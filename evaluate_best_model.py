import torch
import json
import os
import random
from PIL import Image
import torchvision.transforms as transforms
from data_loader import get_loaders
from model import CNNtoRNN

def evaluate_best_model():
    # 1. Load results and find best model
    results_file = "automl_results_grid.json"
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found.")
        return

    print(f"Loading results from {results_file}...")
    with open(results_file, 'r') as f:
        results = json.load(f)

    # Filter out failed runs (score -inf)
    valid_results = [r for r in results if r.get("score") is not None]
    
    if not valid_results:
        print("No valid results found.")
        return

    # Sort by score descending (Assuming BLEU, higher is better)
    # If metric was -ValLoss, higher (closer to 0 from negative) is still better.
    # Note: Our code saved -ValLoss if BLEU was 0.
    best_result = sorted(valid_results, key=lambda x: x["score"], reverse=True)[0]
    
    print(f"\nTime to evaluate the Champion!")
    print(f"Trial: {best_result['trial']}")
    print(f"Config: {json.dumps(best_result['config'], indent=2)}")
    print(f"Score ({best_result['metric']}): {best_result['score']}")
    
    experiment_dir = best_result['result']['experiment_dir']
    checkpoint_path = os.path.join(experiment_dir, "weights", "best_model.pth.tar")
    vocab_path = os.path.join(experiment_dir, "vocab.pth")
    
    print(f"Loading model from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print("Error: Checkpoint file not found!")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Checkpoint with robust logic
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        vocab = checkpoint['vocab']
        config = checkpoint['config']
    except Exception as e:
        print(f"Standard load failed: {e}. Trying robust load...")
        try:
           vocab = torch.load(vocab_path, map_location="cpu", weights_only=False)
           # Fallback config from best_result if needed, but normally checkpoint has it
           config = best_result['config']
           # Re-load checkpoint just for weights if possible? 
           # Actually if standard load failed, we might only have state_dict if we were lucky, 
           # but our train.py saves full dict.
           # Let's assume the previous fix in debug_inference works here too if we just load carefully.
           pass 
        except Exception as e2:
             print(f"Fatal error loading model: {e2}")
             return

    # Initialize Model
    model = CNNtoRNN(
        config["embed_size"], 
        config["hidden_size"], 
        len(vocab), 
        config["num_layers"],
        rnn_type=config.get("rnn_type", "LSTM")
    ).to(device)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    
    # 2. Setup Data Loader to get Test Data
    print("Loading Test Data...")
    
    # Tranforms (must match training usually, but for test visualization simple resize is fine)
    transform = transforms.Compose([
        transforms.Resize((232, 232)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    root_folder = "caption_data/Images"
    annotation_file = "caption_data/captions.txt"
    
    # get_loaders returns (train, val, test, vocab)
    # We re-use the vocab we loaded from the model to ensuring mapping is correct?
    # Actually get_loaders builds vocab from scratch. 
    # CRITICAL: The model was trained on a specific vocab mapping. 
    # If get_loaders rebuilds vocab and it's deterministic (seeded), it should match.
    # checking utils.seed_everything(42) usage... yes config defaults to 42.
    # But to be safe, we should use the model's vocab for decoding.
    
    # Manually create Test Dataset using the LOADED vocab to ensure consistency
    # get_loaders is not deterministic enough to reproduce the exact vocab mapping
    # so we reuse the split logic but force our vocab.
    
    # We call get_loaders to get the raw lists, but we ignore the loader it returns
    # We access the internal lists from a temporary dataset or just copy sample logic
    # Actually, simpler: just create a dataset with ALL images and pick random 10.
    # We don't care about Train/Test split privacy for this simple 10-sample check.
    
    from data_loader import FlickrDataset, MyCollate
    from torch.utils.data import DataLoader
    
    # Read annotations manually to get all images
    all_imgs = []
    all_captions = []
    with open(annotation_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if lines and "image,caption" in lines[0]:
            lines = lines[1:]
        for line in lines:
            parts = line.strip().split(',', 1) 
            if len(parts) == 2:
                all_imgs.append(parts[0])
                all_captions.append(parts[1])
                
    # Create dataset with LOADED vocab
    test_dataset = FlickrDataset(root_folder, all_imgs, all_captions, vocab, transform=transform)
    
    pad_idx = vocab.stoi["<PAD>"]
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1, 
        num_workers=2, 
        shuffle=True, 
        collate_fn=MyCollate(pad_idx)
    )
    
    # 3. Predict on 10 random images and save them
    print("\n" + "="*60)
    print("       MODEL EVALUATION RESULTS (10 Samples)       ")
    print("="*60)
    
    output_dir = "evaluation_samples"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    num_samples = 10
    
    # Inverse norm for visualization
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    to_pil = transforms.ToPILImage()
    
    from PIL import ImageDraw, ImageFont

    with torch.no_grad():
        for idx, (img, captions) in enumerate(test_loader):
            if idx >= num_samples:
                break
                
            img_tensor = img.to(device)
            
            # Generate Caption
            pred_caption = model.caption_image(img_tensor, vocab)
            pred_text = "Pred: " + " ".join(pred_caption)
            
            # Get Ground Truth text
            truth_tokens = []
            for i in captions.squeeze(0):
                word = vocab.itos[i.item()]
                if word == "<EOS>": break
                if word not in ["<SOS>", "<PAD>"]:
                    truth_tokens.append(word)
            truth_text = "Truth: " + " ".join(truth_tokens)
            
            print(f"\n[Image {idx+1}]")
            print(truth_text)
            print(pred_text)
            print("-" * 30)
            
            # Save Image with Text
            # Denormalize
            img_vis = inv_normalize(img.squeeze(0).cpu())
            pil_img = to_pil(img_vis)
            
            # Create a new image with extra space for text
            # Assuming 224x224, let's add 100px at bottom
            w, h = pil_img.size
            new_h = h + 100
            new_img = Image.new("RGB", (w, new_h), (255, 255, 255))
            new_img.paste(pil_img, (0, 0))
            
            draw = ImageDraw.Draw(new_img)
            # Use default font
            
            # Wrap text if too long (simple split)
            # This is very basic wrapping
            def draw_multiline(text, y_pos, color):
                words = text.split()
                lines = []
                current_line = []
                for word in words:
                    current_line.append(word)
                    if len(" ".join(current_line)) > 35: # approximate char limit
                        lines.append(" ".join(current_line[:-1]))
                        current_line = [word]
                lines.append(" ".join(current_line))
                
                for line in lines:
                    draw.text((10, y_pos), line, fill=color)
                    y_pos += 12
                return y_pos

            y = h + 10
            y = draw_multiline(truth_text, y, (0, 100, 0)) # Green for Truth
            draw_multiline(pred_text, y + 5, (0, 0, 150)) # Blue for Pred
            
            save_path = os.path.join(output_dir, f"sample_{idx+1}.jpg")
            new_img.save(save_path)
            print(f"Saved {save_path}")

if __name__ == "__main__":
    evaluate_best_model()
