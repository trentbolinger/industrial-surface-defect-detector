import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# Pull in the validation DataLoader and dataset metadata from preprocess.py
from preprocess import val_loader, full_dataset

CHECKPOINT_PATH = "outputs/best_model.pth"
os.makedirs("outputs", exist_ok=True)

class_names = full_dataset.classes
NUM_CLASSES  = len(class_names)

# Use GPU if available, same as during training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Evaluating on: {device}")

# Rebuild the same model architecture used in train.py
model = models.resnet18(weights=None)  # no pretrained weights — we'll load our own
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

# Load the saved weights from the best checkpoint
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model = model.to(device)

# Switch to eval mode so dropout is disabled and batch norm is frozen
model.eval()

# Run the model over the entire validation set and collect predictions vs. ground truth
all_preds  = []
all_labels = []

with torch.no_grad():  # no gradients needed during evaluation
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs   = model(images)
        predicted = outputs.argmax(dim=1)  # class with the highest score
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

# --- Confusion matrix ---
# Rows = true class, columns = predicted class
# A perfect model has large numbers only on the diagonal
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(8, 7))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, colorbar=True, xticks_rotation=20)
ax.set_title("Confusion Matrix — Validation Set")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png")
print("Confusion matrix saved to outputs/confusion_matrix.png")

# --- Per-class precision, recall, and F1 score ---
# Precision: of all the times the model predicted class X, how often was it right?
# Recall:    of all the true class X images, how many did the model catch?
# F1:        harmonic mean of precision and recall — balances both
print("\nPer-class metrics:")
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
