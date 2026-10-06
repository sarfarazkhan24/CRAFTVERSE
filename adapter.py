import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ============================================================
# ABSTAINITY ADAPTER CONTRACT
# ============================================================

# We point to the penultimate layer of DenseNet121 for OOD detection
FEATURE_LAYER = "features"

def load_model():
    """
    Constructs the DenseNet-121 architecture and loads the trained weights
    from the local model.pt checkpoint.
    """
    model_path = os.path.join(os.path.dirname(__file__), "backend", "model.pt")
    
    # In a real environment, you might be loading this dynamically.
    # Here we assume model.pt is in the backend/ directory relative to the project root.
    if not os.path.exists(model_path):
        # Fallback to local dir if uploaded directly to the same folder as the model
        model_path = "model.pt"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Could not find {model_path}. Please place model.pt next to the adapter.")

    print(f"Loading checkpoint from: {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    
    # Extract num_classes from the checkpoint metadata we saved during training
    num_classes = checkpoint.get("num_classes", 14) # Fallback to 14 if not found
    
    # Reconstruct the architecture exactly as it was during training
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    
    # Load the state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    
    model.eval()
    return model

def preprocess(image_path):
    """
    Loads an image and applies the exact same transformations used during validation.
    """
    # Using the exact mean/std derived from the original training dataset
    MEAN = [0.5407, 0.5407, 0.5407]
    STD = [0.2419, 0.2419, 0.2419]
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image)
    
    # Add batch dimension [1, C, H, W]
    return tensor.unsqueeze(0)

def predict(model, x):
    """
    Runs the forward pass and returns raw logits.
    """
    with torch.no_grad():
        logits = model(x)
    return logits
