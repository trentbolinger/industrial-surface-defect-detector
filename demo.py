import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
from PIL import Image
from torchvision import models, transforms

# ── Model & class setup ──────────────────────────────────────────────────────

CHECKPOINT_PATH = "outputs/best_model.pth"
DATASET_PATH    = "/home/tbolinger/data/neu-dataset/NEU-DET/train/images/"

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

# Use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Rebuild the same ResNet-18 architecture used during training
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))

# Load the saved weights from the best training checkpoint
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model = model.to(device)
model.eval()  # freeze batch norm and disable dropout for inference

# ── Preprocessing ────────────────────────────────────────────────────────────

# Replicate the exact same transforms used in preprocess.py so the model
# sees images in the same format it was trained on
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # NEU images are grayscale; upsample to 3 channels
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── Inference function ───────────────────────────────────────────────────────

def predict(image: Image.Image):
    """Takes a PIL image, runs it through the model, and returns the
    predicted class label and a confidence bar chart."""

    # Apply the same preprocessing the model was trained with
    tensor = preprocess(image).unsqueeze(0).to(device)  # add batch dimension

    # Run the image through the model — no gradient tracking needed
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    # Pick the class with the highest confidence
    predicted_index = int(np.argmax(probabilities))
    confidence      = probabilities[predicted_index] * 100

    # If the model isn't confident enough, reject the image rather than guess
    if confidence < 70.0:
        predicted_label = "Unknown / Not a valid steel surface"
    else:
        predicted_label = CLASS_NAMES[predicted_index]

    result_text = f"{predicted_label}  ({confidence:.1f}% confidence)"

    # Build a horizontal bar chart showing confidence for all 6 classes
    fig, ax = plt.subplots(figsize=(6, 3))
    colors = ["steelblue"] * len(CLASS_NAMES)
    colors[predicted_index] = "tomato"  # highlight the predicted class in red
    ax.barh(CLASS_NAMES, probabilities * 100, color=colors)
    ax.set_xlabel("Confidence (%)")
    ax.set_xlim(0, 100)
    ax.set_title("Model Confidence per Class")
    plt.tight_layout()

    return result_text, fig

# ── Example images ───────────────────────────────────────────────────────────

# Grab one example image from each class folder to show in the Gradio interface.
# If the dataset isn't available on this machine the examples list will just be empty.
examples = []
if os.path.isdir(DATASET_PATH):
    for class_name in CLASS_NAMES:
        class_folder = os.path.join(DATASET_PATH, class_name)
        if os.path.isdir(class_folder):
            files = [f for f in os.listdir(class_folder)
                     if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
            if files:
                examples.append(os.path.join(class_folder, files[0]))

# ── Gradio interface ─────────────────────────────────────────────────────────

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a surface image"),
    outputs=[
        gr.Textbox(label="Predicted Defect"),
        gr.Plot(label="Confidence Scores"),
    ],
    title="Aerospace Defect Detector",
    description=(
        "Upload a steel surface image and the model will classify it into one of "
        "six defect types from the NEU Surface Defect Dataset: "
        "crazing, inclusion, patches, pitted surface, rolled-in scale, or scratches. "
        "Trained on ResNet-18 fine-tuned with PyTorch."
    ),
    examples=examples if examples else None,
    flagging_mode="never",
)

# launch(share=True) generates a public Gradio URL valid for 72 hours
# so the interface can be shared with anyone, even without a local server
demo.launch(share=True)
