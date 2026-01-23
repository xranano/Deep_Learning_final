from train import train
import torch

def verify():
    print("Verifying Attention Model Training Loop...")
    
    config = {
        "learning_rate": 3e-4,
        "batch_size": 2, # Small batch for speed
        "embed_size": 256,
        "hidden_size": 256,
        "num_layers": 1,
        "resize_dim": 232, # Legacy param used in some places maybe?
        "image_size": 224, # Legacy
        "dropout": 0.5,
        "num_epochs": 1,
        "save_model": False,
        "rnn_type": "LSTM",
        "model_type": "attention",
        "attention_dim": 256
    }
    
    # Mock Experiment class or ensure train handles it?
    # train() function creates Experiment internally.
    # It names it based on config.
    
    train(config)
    print("Verification Successful!")

if __name__ == "__main__":
    verify()
