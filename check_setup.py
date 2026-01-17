import sys
import torch
import platform

print("="*40)
print(f"OS: {platform.system()} {platform.release()}")
print(f"Python Version: {sys.version.split()[0]}")
print("="*40)

try:
    print(f"PyTorch Version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available:  {cuda_available}")
    
    if cuda_available:
        print(f"CUDA Version:    {torch.version.cuda}")
        gpu_count = torch.cuda.device_count()
        print(f"GPUs Detected:   {gpu_count}")
        print(f"Current GPU:     {torch.cuda.get_device_name(0)}")
        
        x = torch.tensor([1.0]).cuda()
        print("\nSUCCESS: Tensor successfully loaded to GPU.")
    else:
        print("\nWARNING: PyTorch cannot see your GPU. It will run on CPU (Slow).")
        
except ImportError:
    print("\nERROR: PyTorch is not installed.")

print("="*40)