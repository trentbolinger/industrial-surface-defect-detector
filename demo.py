import os
import copy
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
from PIL import Image
from torchvision import models, transforms

# ── Model & class setup ──────────────────────────────────────────────────────

CHECKPOINT_PATH = "outputs/best_model.pth"
# Path to the dataset — override by setting the NEU_DATASET_PATH environment variable
DATASET_PATH    = os.environ.get("NEU_DATASET_PATH", "data/neu-dataset/NEU-DET/train/images/")

CLASS_NAMES = [
    "crazing",
    "good_steel",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
    "unknown",
]

# The 6 classes that represent actual manufacturing defects (used for DPMO calculation)
DEFECT_CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

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

# ── Quality metrics helpers ───────────────────────────────────────────────────

def make_empty_state():
    """Return a fresh counter dict with all classes zeroed out."""
    return {"total": 0, "unknown_low_conf": 0, **{c: 0 for c in CLASS_NAMES}}

def dpmo_to_sigma(dpmo):
    """Convert a DPMO value to a Six Sigma level using the standard conversion table."""
    if dpmo == 0:
        return "> 6.0"
    elif dpmo <= 3.4:
        return "6.0"
    elif dpmo <= 233:
        return "5.0"
    elif dpmo <= 6_210:
        return "4.0"
    elif dpmo <= 66_807:
        return "3.0"
    elif dpmo <= 308_537:
        return "2.0"
    else:
        return "< 1.0"

def format_dashboard(state):
    """Turn the session counter dict into a Markdown string for the dashboard."""
    total = state["total"]
    if total == 0:
        return "**No images inspected yet.** Submit an image on the Defect Classifier tab to begin."

    good        = state["good_steel"]
    defect_count = sum(state[c] for c in DEFECT_CLASSES)
    rejected    = state["unknown"] + state["unknown_low_conf"]

    # DPMO: defects found per million inspection opportunities
    dpmo  = (defect_count / total) * 1_000_000
    sigma = dpmo_to_sigma(dpmo)

    # First-pass yield: percentage of parts that passed on the first inspection
    fpy   = (good / total) * 100

    lines = [
        "### Session Quality Metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Total Inspected | {total} |",
        f"| Good Steel (Pass) | {good} |",
        f"| Defects Found | {defect_count} |",
        f"| Unknown / Rejected | {rejected} |",
        f"| First-Pass Yield | {fpy:.1f}% |",
        f"| DPMO | {dpmo:,.0f} |",
        f"| Sigma Level | {sigma}σ |",
        "",
        "### Defect Breakdown",
        "",
        "| Defect Class | Count |",
        "|---|---|",
    ]
    for cls in DEFECT_CLASSES:
        lines.append(f"| {cls.replace('_', ' ').title()} | {state[cls]} |")

    return "\n".join(lines)

# ── Inference + state update ─────────────────────────────────────────────────

def classify_and_update(image, state):
    """Run the model on the uploaded image, update session counters, and
    return the prediction result alongside a refreshed dashboard."""

    if image is None:
        return "No image provided.", None, state, format_dashboard(state)

    # Copy the state dict so we don't mutate the previous Gradio state object
    state = copy.copy(state)

    # Apply the same preprocessing the model was trained with
    tensor = preprocess(image).unsqueeze(0).to(device)

    # Run the image through the model — no gradient tracking needed
    with torch.no_grad():
        logits        = model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    # Pick the class with the highest confidence
    predicted_index = int(np.argmax(probabilities))
    confidence      = probabilities[predicted_index] * 100

    # If the model isn't confident enough, reject the image rather than guess
    if confidence < 85.0:
        predicted_label = "Unknown / Not a valid steel surface"
        state["unknown_low_conf"] += 1  # low-confidence rejections tracked separately
    else:
        predicted_label = CLASS_NAMES[predicted_index]
        state[predicted_label] += 1     # increment the matching class counter

    state["total"] += 1

    result_text = f"{predicted_label}  ({confidence:.1f}% confidence)"

    # Build a horizontal bar chart showing confidence for all classes
    fig, ax = plt.subplots(figsize=(6, 3))
    colors = ["steelblue"] * len(CLASS_NAMES)
    colors[predicted_index] = "tomato"  # highlight the predicted class in red
    ax.barh(CLASS_NAMES, probabilities * 100, color=colors)
    ax.set_xlabel("Confidence (%)")
    ax.set_xlim(0, 100)
    ax.set_title("Model Confidence per Class")
    plt.tight_layout()

    # Return prediction outputs, the updated state, and a refreshed dashboard
    return result_text, fig, state, format_dashboard(state)

def reset_session():
    """Clear all session counters and reset the dashboard display."""
    fresh = make_empty_state()
    return fresh, "**Session reset.** Submit an image on the Defect Classifier tab to begin."

# ── Example images ───────────────────────────────────────────────────────────

# Grab one example image from each class folder to populate the examples gallery.
# If the dataset isn't available on this machine the list stays empty.
examples = []
if os.path.isdir(DATASET_PATH):
    for class_name in CLASS_NAMES:
        class_folder = os.path.join(DATASET_PATH, class_name)
        if os.path.isdir(class_folder):
            files = [f for f in os.listdir(class_folder)
                     if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
            if files:
                examples.append(os.path.join(class_folder, files[0]))

# ── Gradio Blocks interface ───────────────────────────────────────────────────

# gr.Blocks is needed here instead of gr.Interface because we require
# both Tabs (two separate views) and State (persistent session counters)
with gr.Blocks(title="Industrial Surface Defect Detector") as demo:

    gr.Markdown("# Industrial Surface Defect Detector")
    gr.Markdown(
        "Upload a steel surface image to classify defects and track session quality metrics. "
        "Trained on ResNet-18 fine-tuned with PyTorch."
    )

    # session_state holds the running counters for this browser session
    session_state = gr.State(make_empty_state())

    with gr.Tabs():

        # ── Tab 1: Defect Classifier ──────────────────────────────────────────
        with gr.Tab("Defect Classifier"):
            img_input       = gr.Image(type="pil", label="Upload a surface image")
            classify_btn    = gr.Button("Classify", variant="primary")
            prediction_out  = gr.Textbox(label="Predicted Defect")
            chart_out       = gr.Plot(label="Confidence Scores")

            # Populate the examples gallery if the dataset is available locally
            if examples:
                gr.Examples(examples=examples, inputs=img_input)

        # ── Tab 2: Quality Dashboard ──────────────────────────────────────────
        with gr.Tab("Quality Dashboard"):
            dashboard_md = gr.Markdown(
                "**No images inspected yet.** Submit an image on the Defect Classifier tab to begin."
            )
            reset_btn = gr.Button("Reset Session", variant="secondary")

    # When the Classify button is clicked, run the model AND refresh the dashboard
    classify_btn.click(
        fn=classify_and_update,
        inputs=[img_input, session_state],
        outputs=[prediction_out, chart_out, session_state, dashboard_md],
    )

    # When the Reset button is clicked, wipe all counters and reset the dashboard text
    reset_btn.click(
        fn=reset_session,
        inputs=[],
        outputs=[session_state, dashboard_md],
    )

# launch(share=True) generates a public Gradio URL valid for 72 hours
# so the interface can be shared with anyone, even without a local server
demo.launch(share=True)
