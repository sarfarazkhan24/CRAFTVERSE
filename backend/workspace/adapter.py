import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv(x)))
        return self.fc(x.view(x.size(0), -1))

FEATURE_LAYER = 'pool'

def load_model():
    model = SimpleCNN()
    model.eval()
    return model

_preprocess_transform = transforms.Compose([
    transforms.ToTensor(),
])

def preprocess(path):
    img = Image.open(path).convert('RGB')
    tensor = _preprocess_transform(img)
    return tensor.unsqueeze(0)  # shape [1, C, H, W]

def predict(model, x):
    with torch.no_grad():
        return model(x)