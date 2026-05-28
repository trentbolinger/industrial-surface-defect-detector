import os
import requests

# Where to save all downloaded images
OUTPUT_DIR = "/home/tbolinger/data/neu-dataset/NEU-DET/train/images/unknown/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# picsum.photos serves random stock photos — it doesn't filter by category,
# so we use different seed ranges to get 40 visually distinct images per group.
# All 240 images land in the same 'unknown/' folder for the defect detector.
CATEGORIES = {
    "cats":      range(1,   41),   # seeds  1 – 40
    "cars":      range(41,  81),   # seeds 41 – 80
    "trees":     range(81,  121),  # seeds 81 – 120
    "food":      range(121, 161),  # seeds 121 – 160
    "people":    range(161, 201),  # seeds 161 – 200
    "buildings": range(201, 241),  # seeds 201 – 240
}

IMAGE_SIZE = 224  # match the size the model was trained on

total = 0
for category, seed_range in CATEGORIES.items():
    print(f"Downloading {category}...")
    for seed in seed_range:
        total += 1
        filename  = f"unknown_{total:03d}.jpg"
        save_path = os.path.join(OUTPUT_DIR, filename)

        # picsum.photos/seed/<seed>/<width>/<height> returns the same image
        # for a given seed, so we get 240 different photos across all seeds
        url      = f"https://picsum.photos/seed/{seed}/{IMAGE_SIZE}/{IMAGE_SIZE}"
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
        else:
            print(f"  Warning: failed to download seed {seed} (HTTP {response.status_code})")

    print(f"  Done — {len(list(seed_range))} images saved")

print(f"\nFinished. {total} images saved to {OUTPUT_DIR}")
