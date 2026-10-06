import os
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np

from abstain.pipeline import AbstainityPipeline
from abstain.fit import fit_reference_set

# 1. Dummy adapter for ResNet18
class Adapter:
    FEATURE_LAYER = "avgpool"
    def load_model(self):
        # We use a pretrained ResNet18
        # weights=models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(pretrained=True)
        model.eval()
        return model
    def preprocess(self, path):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return transform(Image.open(path).convert('RGB')).unsqueeze(0)
    def predict(self, model, x):
        with torch.no_grad():
            out = model(x)
        return torch.softmax(out, dim=1)

adapter = Adapter()
model = adapter.load_model()
device = "cpu"
model.to(device)

# 2. Create fake reference data
os.makedirs("ref_data/dog", exist_ok=True)
os.makedirs("ref_data/cat", exist_ok=True)
for i in range(15):
    Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)).save(f"ref_data/dog/img_{i}.jpg")
    Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)).save(f"ref_data/cat/img_{i}.jpg")

# 3. Fit
config_path = "test_config.json"
fit_reference_set(adapter, model, "ref_data", config_path)

# 4. Run tests
pipeline = AbstainityPipeline(config_path)

Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)).save("clean.jpg")
Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)).save("alien.jpg")

print("\n--- CLEAN IMAGE ---")
print(pipeline.predict(adapter, model, adapter.preprocess("clean.jpg")))

print("\n--- ALIEN IMAGE ---")
print(pipeline.predict(adapter, model, adapter.preprocess("alien.jpg")))
