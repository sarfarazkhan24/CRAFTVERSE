import torch

def load_model():
    print("Loading dummy model...")
    # Return a dummy model
    return torch.nn.Linear(10, 2)

def preprocess(path):
    print(f"Preprocessing image at {path}")
    return torch.randn(1, 10)

def predict(model, x):
    print("Running prediction...")
    return model(x)

FEATURE_LAYER = "dummy_features"
