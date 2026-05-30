import os
import torch
import torch.nn as nn
from torchvision import models

# Pull in the DataLoaders and full dataset built in preprocess.py
from preprocess import train_loader, val_loader, full_dataset

# Where to save the best model checkpoint
os.makedirs("outputs", exist_ok=True)
CHECKPOINT_PATH = "outputs/best_model.pth"

LEARNING_RATE = 0.001
NUM_EPOCHS = 30
NUM_CLASSES = len(full_dataset.classes)  # 8 classes: 6 defects + unknown + good_steel

# Use GPU if one is available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# Load ResNet-18 with weights pretrained on ImageNet
# Pretrained weights give us a head start — the model already understands
# basic visual features like edges and textures
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Swap out the final fully-connected layer so it outputs 7 scores (6 defects + unknown)
# instead of the original 1000 ImageNet classes
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

# Move the model's parameters to the chosen device
model = model.to(device)

# Cross-entropy loss is the standard choice for multi-class classification
loss_fn = nn.CrossEntropyLoss()

# Adam optimizer updates the model weights during backpropagation
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# StepLR reduces the learning rate by a factor of 0.1 every 10 epochs
# e.g. 0.001 -> 0.0001 at epoch 10 -> 0.00001 at epoch 20
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

best_val_accuracy = 0.0  # track the highest validation accuracy seen so far

for epoch in range(1, NUM_EPOCHS + 1):

    # --- Training phase ---
    model.train()  # puts the model in training mode (enables dropout, batch norm updates)
    running_loss = 0.0

    for images, labels in train_loader:
        # Move this batch to the same device as the model
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()          # clear gradients from the previous step
        outputs = model(images)        # forward pass — get raw class scores
        loss = loss_fn(outputs, labels)  # compare predictions to ground truth
        loss.backward()               # backprop — compute gradients
        optimizer.step()              # update weights

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    # --- Validation phase ---
    model.eval()  # puts the model in eval mode (disables dropout, freezes batch norm)
    correct = 0
    total = 0

    with torch.no_grad():  # no need to track gradients during validation
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = outputs.argmax(dim=1)  # pick the class with the highest score
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_accuracy = correct / total

    # Step the scheduler at the end of each epoch to update the learning rate
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]

    print(f"Epoch {epoch:>2}/{NUM_EPOCHS} | "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Accuracy: {val_accuracy:.4f} | "
          f"LR: {current_lr:.6f}")

    # Save the model whenever validation accuracy beats the previous best
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(f"           -> New best model saved ({val_accuracy:.4f})")

print(f"\nTraining complete. Best validation accuracy: {best_val_accuracy:.4f}")
print(f"Best model saved to {CHECKPOINT_PATH}")
