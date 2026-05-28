import os
import random
import matplotlib.pyplot as plt
from PIL import Image

# Path to the folder that contains one sub-folder per defect class
DATASET_PATH = "/home/tbolinger/data/neu-dataset/NEU-DET/train/images/"

# Where to save the grid image
OUTPUT_PATH = "outputs/sample_grid.png"

# Create the outputs folder if it doesn't exist yet
os.makedirs("outputs", exist_ok=True)

# Collect all class names by listing the sub-folders inside the dataset path
class_names = sorted(
    name for name in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, name))
)

# Pick one random image file from each class folder
samples = []
for class_name in class_names:
    class_folder = os.path.join(DATASET_PATH, class_name)
    image_files = [
        f for f in os.listdir(class_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]
    chosen_file = random.choice(image_files)
    chosen_path = os.path.join(class_folder, chosen_file)
    samples.append((class_name, chosen_path))

# Set up a 2-row by 3-column grid of subplots
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# Flatten the 2D array of axes into a simple list so we can loop over it easily
axes = axes.flatten()

# Load and display each sampled image in its subplot
for ax, (class_name, image_path) in zip(axes, samples):
    img = Image.open(image_path)
    ax.imshow(img, cmap="gray" if img.mode == "L" else None)
    ax.set_title(class_name, fontsize=13)
    ax.axis("off")  # hide axis tick marks so only the image shows

plt.suptitle("NEU-DET — One Random Sample per Defect Class", fontsize=15, y=1.02)
plt.tight_layout()

# Save the finished grid to disk
plt.savefig(OUTPUT_PATH, bbox_inches="tight")
print(f"Grid saved to {OUTPUT_PATH}")
