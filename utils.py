import torch
import numpy as np
import random
import os
import json
from datetime import datetime
from tqdm import tqdm

def seed_everything(seed=42):
    """Sets the seed for reproducibility across all libraries."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Experiment:
    """
    Manages experiment directories, configurations, and logging.
    Structure:
    experiments/
        2024-01-17_18-30-00_ResNet101_LSTM/
            config.json
            logs/
            weights/
                checkpoint_epoch_1.pth.tar
    """
    def __init__(self, name, config, root="experiments"):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.name = f"{timestamp}_{name}"
        self.dir = os.path.join(root, self.name)
        self.weights_dir = os.path.join(self.dir, "weights")
        self.logs_dir = os.path.join(self.dir, "logs")
        
        os.makedirs(self.weights_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        self.save_config(config)
        
        print(f"[Experiment] Initialized: {self.dir}")
        print(f"[Experiment] Config saved.")

    def save_config(self, config):
        with open(os.path.join(self.dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)
            
    def log_metric(self, name, value, step, writer=None):
        """Logs a metric to console and optional TensorBoard writer."""
        if writer:
            writer.add_scalar(name, value, step)
            
    def get_checkpoint_path(self, epoch):
        return os.path.join(self.weights_dir, f"checkpoint_epoch_{epoch}.pth.tar")
    
def print_examples(model, device, dataset, vocab, n=2):
    """Prints a few example captions (prediction vs truth)"""
    model.eval()
    print("\n--- Example Predictions ---")
    indices = np.random.choice(len(dataset), n, replace=False)
    
    for idx in indices:
        img, caption_tensor = dataset[idx]
        img = img.unsqueeze(0).to(device)
        
        truth = []
        for i in caption_tensor:
            word = vocab.itos[i.item()]
            if word == "<EOS>": break
            if word not in ["<SOS>", "<PAD>"]: truth.append(word)
        
        with torch.no_grad():
            output = model.caption_image(img, vocab)
        
        print(f"Truth: {' '.join(truth)}")
        print(f"Pred : {' '.join(output)}")
        print("---------------------------")
    model.train()

def evaluate_bleu(loader, model, device, vocab):
    """
    Calculates BLEU-4 score over the dataset.
    Note: For strict academic BLEU, we need 5 references per image.
    This implementation compares Prediction vs 1 Target (the one in the batch).
    It works as a relative metic for tracking progress.
    """
    from torchmetrics import BLEUScore
    
    print("=> Calculating BLEU Score (this might take a while)...")
    metric = BLEUScore(n_gram=4, smooth=True)
    model.eval()
    
    preds = []
    targets = []
    
    limit_batches = 50 
    
    with torch.no_grad():
        for idx, (imgs, captions) in enumerate(tqdm(loader, desc="BLEU Eval", leave=False)):
            if idx > limit_batches: break
            
            imgs = imgs.to(device)
            
            for i in range(imgs.shape[0]):
                img = imgs[i].unsqueeze(0)
                
                generated = model.caption_image(img, vocab) 
                preds.append(" ".join(generated))
                
                truth_words = []
                for token_idx in captions[:, i]:
                    word = vocab.itos[token_idx.item()]
                    if word == "<EOS>": break
                    if word not in ["<SOS>", "<PAD>"]: truth_words.append(word)
                
                targets.append([" ".join(truth_words)])
    
    score = metric(preds, targets)
    print(f"=> BLEU-4 Score: {score.item():.4f}")
    model.train()
    return score.item()
