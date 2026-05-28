import os
import matplotlib.pyplot as plt

# Path to the dataset — override by setting the NEU_DATASET_PATH environment variable
DATASET_PATH = os.environ.get("NEU_DATASET_PATH", "data/neu-dataset/NEU-DET/train/images/")

# Where to save the bar chart — created below if it doesn't already exist
OUTPUT_DIR = "outputs"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "class_distribution.png")

# Create the outputs folder if it doesn't exist yet
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Count how many image files are in each class sub-folder
class_counts = {}
for class_name in sorted(os.listdir(DATASET_PATH)):
    class_folder = os.path.join(DATASET_PATH, class_name)
    if not os.path.isdir(class_folder):
        continue  # skip any stray files at the top level
    image_files = [
        f for f in os.listdir(class_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]
    class_counts[class_name] = len(image_files)

# Print the count for each defect class
print("Image counts per class:")
for class_name, count in class_counts.items():
    print(f"  {class_name}: {count}")

# Plot a bar chart showing the class distribution
classes = list(class_counts.keys())
counts = list(class_counts.values())

plt.figure(figsize=(10, 5))
plt.bar(classes, counts, color="steelblue")
plt.title("NEU-DET Training Set — Images per Defect Class")
plt.xlabel("Defect Class")
plt.ylabel("Number of Images")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()

# Save the chart to disk instead of opening an interactive window
plt.savefig(OUTPUT_FILE)
print(f"\nChart saved to {OUTPUT_FILE}")
