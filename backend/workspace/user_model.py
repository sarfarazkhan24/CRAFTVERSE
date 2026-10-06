import torch
import torch.nn as nn
from torchvision import models
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os


# -----------------------------------------
# Device
# -----------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# -----------------------------------------
# Dataset
# -----------------------------------------

mean = [0.5407, 0.5407, 0.5407]
std = [0.2419, 0.2419, 0.2419]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])


class ChestXrayDataset(Dataset):

    def __init__(self, csv_file, image_dir):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir

        self.labels = sorted(self.df["label"].unique())
        self.label_to_id = {
            label: i for i, label in enumerate(self.labels)
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        path = os.path.join(
            self.image_dir,
            row["filename"]
        )

        image = Image.open(path).convert("RGB")
        image = transform(image)

        label = self.label_to_id[row["label"]]

        return image, label


dataset = ChestXrayDataset(
    "split_manifest.csv",
    "images"
)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)


# -----------------------------------------
# Load DenseNet-121
# -----------------------------------------

model = models.densenet121(weights="DEFAULT")

model.classifier = nn.Linear(
    model.classifier.in_features,
    15
)

model = model.to(device)

model.eval()


# -----------------------------------------
# GPU forward pass
# -----------------------------------------

images, labels = next(iter(loader))

images = images.to(device)

with torch.no_grad():
    outputs = model(images)


print("Input shape:", images.shape)
print("Output shape:", outputs.shape)
print("GPU memory used:",
      round(torch.cuda.memory_allocated() / 1024**2, 2),
      "MB")