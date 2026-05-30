import os
import subprocess
import zipfile
import pandas as pd

# Path to the Severstal sample submission CSV — each row contains an image filename
CSV_PATH   = "/home/tbolinger/data/severstal/sample_submission.csv"

# Where to save the downloaded images
OUTPUT_DIR = "/home/tbolinger/data/neu-dataset/NEU-DET/train/images/good_steel/"

COMPETITION = "severstal-steel-defect-detection"
NUM_IMAGES  = 240

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read the CSV — the ImageId_ClassId column contains entries like "abc123.jpg_1",
# so we split on the last underscore to get just the image filename
df = pd.read_csv(CSV_PATH)
all_filenames = df["ImageId"].unique().tolist()

# Take only the first 240 unique image filenames
filenames = all_filenames[:NUM_IMAGES]
print(f"Found {len(all_filenames)} unique images — downloading first {NUM_IMAGES}.")

downloaded = 0
for i, filename in enumerate(filenames, 1):
    save_path = os.path.join(OUTPUT_DIR, filename)

    # Skip images that were already downloaded in a previous run
    if os.path.exists(save_path):
        if i % 10 == 0:
            print(f"Progress: {i}/{NUM_IMAGES} — {filename} already exists, skipping.")
        continue

    # Use the kaggle CLI to download one specific file from the competition
    # The -f flag targets a single file; -p sets the output directory
    result = subprocess.run(
        [
            "kaggle", "competitions", "download",
            "-c", COMPETITION,
            "-f", f"train_images/{filename}",
            "-p", OUTPUT_DIR,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"  Warning: failed to download {filename} — {result.stderr.strip()}")
        continue

    # The kaggle CLI sometimes wraps individual files in a zip — extract and clean up if so
    zip_path = os.path.join(OUTPUT_DIR, filename + ".zip")
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(OUTPUT_DIR)
        os.remove(zip_path)

    downloaded += 1

    # Print a progress update every 10 images
    if i % 10 == 0:
        print(f"Progress: {i}/{NUM_IMAGES} images processed ({downloaded} downloaded).")

print(f"\nDone. {downloaded} new images saved to {OUTPUT_DIR}")
