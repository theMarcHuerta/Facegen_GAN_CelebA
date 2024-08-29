import os
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Define the path where the CelebA dataset is located
data_dir = 'archive/img_align_celeba/img_align_celeba'

# Define a transformation to preprocess the images
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128
    transforms.ToTensor(),          # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

# Load the dataset
dataset = ImageFolder(root=data_dir, transform=transform)

# Define a DataLoader to read images in batches
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

# Check the number of images
print(f'Total number of images: {len(dataset)}')
