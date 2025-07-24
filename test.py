import torch

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Number of GPUs:", torch.cuda.device_count())
    print("GPU Name:", torch.cuda.get_device_name(0))

    # Try allocating a tensor on GPU
    try:
        device = torch.device("cuda")
        x = torch.randn(3, 3).to(device)
        print("Tensor successfully moved to GPU:")
        print(x)
    except Exception as e:
        print("Error while using GPU:", e)
else:
    print("CUDA is not available. You're using CPU.")
