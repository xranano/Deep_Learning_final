import torch
import sys
import os

print("Verifying imports...")
try:
    from data_loader import Vocabulary, FlickrDataset, get_loaders
    print("data_loader imported successfully.")
except Exception as e:
    print(f"Error importing data_loader: {e}")

try:
    from model import EncoderCNN, DecoderRNN, CNNtoRNN
    print("model imported successfully.")
except Exception as e:
    print(f"Error importing model: {e}")

try:
    import train
    print("train script imported successfully.")
except Exception as e:
    print(f"Error importing train: {e}")

try:
    import inference
    print("inference script imported successfully.")
except Exception as e:
    print(f"Error importing inference: {e}")

print("Verification complete.")
