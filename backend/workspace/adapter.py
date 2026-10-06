import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

def find_feature_layer(model):
    last_conv_name = None
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.AdaptiveAvgPool2d, nn.MaxPool2d, nn.Linear)):
            if not isinstance(module, nn.Linear):
                last_conv_name = name
    if hasattr(model, 'features'):
        return 'features'
    return last_conv_name

FEATURE_LAYER = None # Will be populated dynamically or you can set it if known

def load_model():
    
    model_path = os.path.join(os.path.dirname(__file__), "model.pt")
    try:
        model = torch.jit.load(model_path, map_location="cpu")
    except Exception:
        model = torch.load(model_path, map_location="cpu")

    model.eval()
    
    global FEATURE_LAYER
    FEATURE_LAYER = find_feature_layer(model)
    print(f"Auto-detected FEATURE_LAYER: {FEATURE_LAYER}")
    return model

def preprocess(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image)
    return tensor.unsqueeze(0)

def predict(model, x):
    with torch.no_grad():
        out = model(x)
    
    # Check if out is a tuple (some models return multiple things)
    if isinstance(out, tuple):
        out = out[0]
        
    # If the output is logits (can have negative values or sum != 1), apply softmax
    if out.min() < 0 or not torch.allclose(out.sum(dim=1), torch.tensor(1.0), atol=1e-3):
        return torch.softmax(out, dim=1)
    return out
