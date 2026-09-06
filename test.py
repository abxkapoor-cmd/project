import torch
import torchvision
print(torch.__version__)
print(torchvision.__version__)
print("GPU available:", torch.cuda.is_available())