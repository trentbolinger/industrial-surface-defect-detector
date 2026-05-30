import os
import torch
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms

# Path to the dataset — override by setting the NEU_DATASET_PATH environment variable
DATASET_PATH = os.environ.get("NEU_DATASET_PATH", "data/neu-dataset/NEU-DET/train/images/")

BATCH_SIZE = 32

# Define the sequence of transforms applied to every image before it enters the model:
#   - Resize to 224x224 so all images are the same size
#   - Convert to RGB in case any images are grayscale (ensures 3 color channels)
#   - Convert the image to a PyTorch tensor (scales pixel values from 0-255 to 0.0-1.0)
#   - Normalize each channel using ImageNet mean and std (helps the model train faster)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # converts grayscale to 3-channel RGB
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Load the full dataset — ImageFolder automatically assigns a label to each image
# based on which sub-folder (class) it lives in
full_dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform)

# Calculate how many images go into each split
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

# Randomly split the dataset into training and validation sets
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

print(f"Total images  : {len(full_dataset)}")
print(f"Training      : {len(train_dataset)}")
print(f"Validation    : {len(val_dataset)}")
print(f"Classes       : {full_dataset.classes}")

# Wrap each split in a DataLoader — this handles batching and shuffling automatically
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

# Grab one batch from the training loader and print its shape as a sanity check
images, labels = next(iter(train_loader))
print(f"\nOne batch — images shape : {images.shape}")  # expect [32, 3, 224, 224]
print(f"One batch — labels shape : {labels.shape}")   # expect [32]
