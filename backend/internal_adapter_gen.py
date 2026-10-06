import json

def generate_adapter(model_type: str, preset: str, class_name: str = None, kwargs_dict: dict = None, custom_mean=None, custom_std=None, custom_size=None):
    
    # Preprocessing presets
    if preset == "imagenet":
        mean = "[0.485, 0.456, 0.406]"
        std = "[0.229, 0.224, 0.225]"
        size = "(224, 224)"
        channels = 3
    elif preset == "mnist":
        mean = "[0.5]"
        std = "[0.5]"
        size = "(28, 28)"
        channels = 1
    elif preset == "clinsure":
        mean = "[0.5407, 0.5407, 0.5407]"
        std = "[0.2419, 0.2419, 0.2419]"
        size = "(224, 224)"
        channels = 3
    else:
        mean = str(custom_mean) if custom_mean else "[0.5, 0.5, 0.5]"
        std = str(custom_std) if custom_std else "[0.5, 0.5, 0.5]"
        size = str(custom_size) if custom_size else "(224, 224)"
        channels = 3
        
    convert_str = "convert('RGB')" if channels == 3 else "convert('L')"

    # Model Loading
    if model_type == "weights_and_script":
        kwargs_str = ", ".join([f"{k}={repr(v)}" for k, v in (kwargs_dict or {}).items()])
        load_code = f"""
    import sys
    import os
    # Add workspace to path to allow importing model script
    sys.path.insert(0, os.path.dirname(__file__))
    from user_model import {class_name}
    
    model = {class_name}({kwargs_str})
    model_path = os.path.join(os.path.dirname(__file__), "weights.pt")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
"""
    else:
        # full_model
        load_code = """
    model_path = os.path.join(os.path.dirname(__file__), "model.pt")
    try:
        model = torch.jit.load(model_path, map_location="cpu")
    except Exception:
        model = torch.load(model_path, map_location="cpu")
"""

    code = f"""import os
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
    {load_code}
    model.eval()
    
    global FEATURE_LAYER
    FEATURE_LAYER = find_feature_layer(model)
    print(f"Auto-detected FEATURE_LAYER: {{FEATURE_LAYER}}")
    return model

def preprocess(image_path):
    transform = transforms.Compose([
        transforms.Resize({size}),
        transforms.ToTensor(),
        transforms.Normalize({mean}, {std})
    ])
    image = Image.open(image_path).{convert_str}
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
"""
    return code
