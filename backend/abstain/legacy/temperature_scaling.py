import os
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import log_loss


# =========================================================
# SETTINGS
# =========================================================

CSV_FILE = "split_manifest.csv"
IMAGE_DIR = "images"
MODEL_FILE = "model.pt"

BATCH_SIZE = 16
NUM_BINS = 10

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# =========================================================
# TRANSFORM
# =========================================================

mean = [0.5407, 0.5407, 0.5407]
std = [0.2419, 0.2419, 0.2419]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])


# =========================================================
# DATASET
# =========================================================

class ChestXrayDataset(Dataset):

    def __init__(self, csv_file, image_dir, split):

        df = pd.read_csv(csv_file)

        self.df = df[
            df["split"] == split
        ].reset_index(drop=True)

        self.image_dir = image_dir

        self.labels = sorted(
            df["label"].unique()
        )

        self.label_to_id = {
            label: i
            for i, label in enumerate(self.labels)
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

        label = self.label_to_id[
            row["label"]
        ]

        return image, label


# =========================================================
# VALIDATION DATA
# =========================================================

dataset = ChestXrayDataset(
    CSV_FILE,
    IMAGE_DIR,
    "val"
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Validation images:", len(dataset))
print("Number of classes:", len(dataset.labels))


# =========================================================
# MODEL
# =========================================================

model = models.densenet121(
    weights=None
)

model.classifier = nn.Linear(
    model.classifier.in_features,
    len(dataset.labels)
)

checkpoint = torch.load(
    MODEL_FILE,
    map_location=device,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()

print(
    "Loaded checkpoint epoch:",
    checkpoint.get("epoch")
)


# =========================================================
# COLLECT LOGITS
# =========================================================

logits_list = []
labels_list = []

with torch.no_grad():

    for images, labels in loader:

        images = images.to(device)

        outputs = model(images)

        logits_list.append(
            outputs.cpu()
        )

        labels_list.append(
            labels
        )

logits = torch.cat(
    logits_list
)

labels = torch.cat(
    labels_list
)

print("Logits shape:", logits.shape)


# =========================================================
# ECE FUNCTION
# =========================================================

def calculate_ece(logits, labels, n_bins=10):

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    confidences, predictions = torch.max(
        probabilities,
        dim=1
    )

    accuracies = (
        predictions == labels
    ).float()

    ece = torch.tensor(0.0)

    bin_boundaries = torch.linspace(
        0,
        1,
        n_bins + 1
    )

    for i in range(n_bins):

        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]

        if i == 0:
            mask = (
                (confidences >= lower) &
                (confidences <= upper)
            )
        else:
            mask = (
                (confidences > lower) &
                (confidences <= upper)
            )

        if mask.sum() > 0:

            bin_accuracy = accuracies[mask].mean()

            bin_confidence = confidences[mask].mean()

            bin_fraction = (
                mask.float().mean()
            )

            ece += (
                bin_fraction *
                torch.abs(
                    bin_accuracy -
                    bin_confidence
                )
            )

    return ece.item()


# =========================================================
# NLL
# =========================================================

criterion = nn.CrossEntropyLoss()

nll_before = criterion(
    logits,
    labels
).item()


# =========================================================
# ECE BEFORE
# =========================================================

ece_before = calculate_ece(
    logits,
    labels,
    NUM_BINS
)

print("\n========== BEFORE CALIBRATION ==========")

print(
    f"NLL: {nll_before:.6f}"
)

print(
    f"ECE: {ece_before:.6f}"
)


# =========================================================
# TEMPERATURE
# =========================================================

temperature = nn.Parameter(
    torch.ones(1) * 1.0
)

optimizer = torch.optim.LBFGS(
    [temperature],
    lr=0.01,
    max_iter=50
)


def closure():

    optimizer.zero_grad()

    loss = criterion(
        logits / temperature,
        labels
    )

    loss.backward()

    return loss


optimizer.step(closure)


T = temperature.item()

print("\n========== TEMPERATURE ==========")

print(
    f"Optimal Temperature: {T:.6f}"
)


# =========================================================
# AFTER CALIBRATION
# =========================================================

scaled_logits = (
    logits / T
)

nll_after = criterion(
    scaled_logits,
    labels
).item()

ece_after = calculate_ece(
    scaled_logits,
    labels,
    NUM_BINS
)

print("\n========== AFTER CALIBRATION ==========")

print(
    f"NLL: {nll_after:.6f}"
)

print(
    f"ECE: {ece_after:.6f}"
)


# =========================================================
# RELIABILITY DATA
# =========================================================

def reliability_data(
    logits,
    labels,
    n_bins=10
):

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    confidences, predictions = torch.max(
        probabilities,
        dim=1
    )

    accuracies = (
        predictions == labels
    ).float()

    bin_boundaries = torch.linspace(
        0,
        1,
        n_bins + 1
    )

    bin_conf = []
    bin_acc = []

    for i in range(n_bins):

        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]

        if i == 0:
            mask = (
                (confidences >= lower) &
                (confidences <= upper)
            )
        else:
            mask = (
                (confidences > lower) &
                (confidences <= upper)
            )

        if mask.sum() > 0:

            bin_conf.append(
                confidences[mask].mean().item()
            )

            bin_acc.append(
                accuracies[mask].mean().item()
            )

    return bin_conf, bin_acc


before_conf, before_acc = reliability_data(
    logits,
    labels,
    NUM_BINS
)

after_conf, after_acc = reliability_data(
    scaled_logits,
    labels,
    NUM_BINS
)


# =========================================================
# RELIABILITY DIAGRAM
# =========================================================

plt.figure(figsize=(7, 7))

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect calibration"
)

plt.plot(
    before_conf,
    before_acc,
    marker="o",
    label="Before calibration"
)

plt.plot(
    after_conf,
    after_acc,
    marker="o",
    label="After calibration"
)

plt.xlabel("Confidence")
plt.ylabel("Accuracy")

plt.title(
    "Reliability Diagram - Temperature Scaling"
)

plt.legend()

plt.grid()

plt.tight_layout()

plt.savefig(
    "reliability_diagram.png",
    dpi=200
)

plt.close()


# =========================================================
# SAVE RESULTS
# =========================================================

results = {

    "checkpoint_epoch":
        checkpoint.get("epoch"),

    "temperature":
        float(T),

    "nll_before":
        float(nll_before),

    "nll_after":
        float(nll_after),

    "ece_before":
        float(ece_before),

    "ece_after":
        float(ece_after),

    "validation_images":
        len(dataset),

    "num_bins":
        NUM_BINS
}


with open(
    "temperature.json",
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


print("\n================================")
print("Saved: temperature.json")
print("Saved: reliability_diagram.png")
print("================================")