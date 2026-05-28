import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models

# Pull in the DataLoaders and dataset metadata from preprocess.py
from preprocess import train_loader, val_loader, full_dataset

os.makedirs("outputs", exist_ok=True)

LEARNING_RATE = 0.001
NUM_EPOCHS    = 30
NUM_CLASSES   = len(full_dataset.classes)  # 7 classes: 6 defects + unknown

# Use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# Build the same ResNet-18 architecture used in train.py
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(device)

loss_fn   = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# StepLR drops the learning rate by 10x every 10 epochs, matching train.py
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Lists to record one value per epoch so we can plot them afterwards
train_losses   = []
val_accuracies = []

for epoch in range(1, NUM_EPOCHS + 1):

    # --- Training phase ---
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)

    # --- Validation phase ---
    model.eval()
    correct = 0
    total   = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs   = model(images)
            predicted = outputs.argmax(dim=1)
            correct  += (predicted == labels).sum().item()
            total    += labels.size(0)

    val_acc = correct / total
    val_accuracies.append(val_acc)

    scheduler.step()

    print(f"Epoch {epoch:>2}/{NUM_EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

# --- Plotting ---
# Draw train loss on the left y-axis and validation accuracy on the right y-axis
# so both curves share the same x-axis (epochs) despite having different scales
epochs = range(1, NUM_EPOCHS + 1)

fig, ax1 = plt.subplots(figsize=(10, 5))

# Left axis — training loss (lower is better)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Training Loss", color="steelblue")
ax1.plot(epochs, train_losses, color="steelblue", linewidth=2, label="Train Loss")
ax1.tick_params(axis="y", labelcolor="steelblue")

# Right axis — validation accuracy (higher is better)
ax2 = ax1.twinx()
ax2.set_ylabel("Validation Accuracy", color="tomato")
ax2.plot(epochs, val_accuracies, color="tomato", linewidth=2, label="Val Accuracy")
ax2.tick_params(axis="y", labelcolor="tomato")
ax2.set_ylim(0, 1)

# Combine both legends into one box
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

plt.title("Training Curves — Loss & Validation Accuracy over 30 Epochs")
fig.tight_layout()
plt.savefig("outputs/training_curves.png")
print("\nChart saved to outputs/training_curves.png")
