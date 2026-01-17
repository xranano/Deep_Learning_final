import torch
import torchvision.transforms as transforms
from PIL import Image
from model import CNNtoRNN
import os

def load_inference_model(checkpoint_path, device):
    """
    Loads model, vocab, and transform from a specific checkpoint.
    """
    print(f"=> Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint['config']
    vocab = checkpoint['vocab']
    state_dict = checkpoint['state_dict']
    
    print(f"   Experiment: {config['experiment_name']}")
    print(f"   Epoch: {checkpoint['epoch']}")
    print(f"   Vocab size: {len(vocab)}")

    model = CNNtoRNN(
        embed_size=config['embed_size'],
        hidden_size=config['hidden_size'],
        vocab_size=len(vocab),
        num_layers=config['num_layers']
    ).to(device)
    
    model.load_state_dict(state_dict)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((config['resize_dim'], config['resize_dim'])),
        transforms.CenterCrop((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    return model, vocab, transform

def generate_caption(image_path, model, vocab, transform, device, max_length=50):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    print(f"=> Generating caption for {image_path}...")
    with torch.no_grad():
        output = model.caption_image(image_tensor, vocab, max_length=max_length)
        
        sentence = " ".join(output)
        print(f"\nOUTPUT: {sentence}\n")
        return sentence

if __name__ == "__main__":
    CHECKPOINT_FILE = "path/to/your/checkpoint.pth.tar" 
    IMAGE_FILE = "test_image.jpg"
    
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.zeros(1).to(device)
            try:
                torch.nn.LSTM(10, 10).to(device)
            except Exception:
                print("[WARNING] Standard GPU LSTM failed. Disabling CuDNN for inference...")
                torch.backends.cudnn.enabled = False
                torch.nn.LSTM(10, 10).to(device)
        else:
            device = torch.device("cpu")
    except Exception:
        print("[WARNING] CUDA failed completely. Falling back to CPU.")
        device = torch.device("cpu")
    
        print("Model loaded successfully.")

    if not os.path.exists(image_path):
        print(f"Image {image_path} not found. Using a random tensor for demo.")
        image_tensor = torch.rand(3, 224, 224)
        img = image_tensor
    else:
        img = Image.open(image_path).convert("RGB")
        img = transform(img)
    
    img = img.to(device)
    
    print("Generating caption...")
    caption = model.caption_image(img, vocab, device=device)
    
    print("Caption:")
    print(" ".join(caption))

if __name__ == "__main__":
    main()
