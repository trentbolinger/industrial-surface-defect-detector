from PIL import Image
import matplotlib.pyplot as plt

# Swap this path out for the actual image you want to inspect
IMAGE_PATH = "path/to/your/image.jpg"

# Open the image file using Pillow
img = Image.open(IMAGE_PATH)

# Print the image dimensions (width x height in pixels) and color mode (e.g. RGB, L for grayscale)
print(f"Size: {img.size[0]}w x {img.size[1]}h pixels")
print(f"Mode: {img.mode}")

# Display the image in a matplotlib window
plt.imshow(img)
plt.title(IMAGE_PATH)
plt.axis("off")  # hide the axis tick marks so only the image shows
plt.show()
