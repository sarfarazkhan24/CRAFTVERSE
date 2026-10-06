import os
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


# =========================================================
# SETTINGS
# =========================================================

CSV_FILE = "split_manifest.csv"
IMAGE_DIR = "images"
MODEL_FILE = "model.pt"

BATCH_SIZE = 16
MC_PASSES = 10

DROPOUT_RATE = 0.20

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

        image = Image.open(
            path
        ).convert("RGB")

        image = transform(image)

        label = self.label_to_id[
            row["label"]
        ]

        return image, label, row["filename"]


# =========================================================
# DATA
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
print("MC passes:", MC_PASSES)


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


# =========================================================
# DROPOUT MODULE
# =========================================================

dropout = nn.Dropout(
    p=DROPOUT_RATE
).to(device)


# =========================================================
# PENULTIMATE FEATURES
# =========================================================

def get_logits_with_dropout(images):

    features = model.features(images)

    features = torch.relu(features)

    features = torch.nn.functional.adaptive_avg_pool2d(
        features,
        (1, 1)
    )

    features = torch.flatten(
        features,
        1
    )

    # Artificial dropout during MC sampling
    features = dropout(features)

    logits = model.classifier(
        features
    )

    return logits


# =========================================================
# MC-DROPOUT PREDICTIONS
# =========================================================

all_mean_probs = []
all_variance = []
all_entropy = []
all_predictions = []
all_labels = []
all_filenames = []

model.eval()

with torch.no_grad():

    for batch_idx, (
        images,
        labels,
        filenames
    ) in enumerate(loader):

        images = images.to(device)

        mc_probs = []

        for _ in range(MC_PASSES):

            logits = get_logits_with_dropout(
                images
            )

            probs = torch.softmax(
                logits,
                dim=1
            )

            mc_probs.append(
                probs.cpu()
            )

        mc_probs = torch.stack(
            mc_probs
        )

        # Mean probability
        mean_probs = mc_probs.mean(
            dim=0
        )

        # Variance across MC passes
        variance = mc_probs.var(
            dim=0
        ).mean(
            dim=1
        )

        # Predictive entropy
        entropy = -(
            mean_probs *
            torch.log(
                mean_probs + 1e-8
            )
        ).sum(
            dim=1
        )

        predictions = torch.argmax(
            mean_probs,
            dim=1
        )

        all_mean_probs.extend(
            mean_probs.numpy()
        )

        all_variance.extend(
            variance.numpy()
        )

        all_entropy.extend(
            entropy.numpy()
        )

        all_predictions.extend(
            predictions.numpy()
        )

        all_labels.extend(
            labels.numpy()
        )

        all_filenames.extend(
            filenames
        )

        if (batch_idx + 1) % 20 == 0:

            print(
                f"Processed "
                f"{min((batch_idx + 1) * BATCH_SIZE, len(dataset))}"
                f"/{len(dataset)}"
            )


# =========================================================
# RESULTS
# =========================================================

mean_probs = np.array(
    all_mean_probs
)

variance = np.array(
    all_variance
)

entropy = np.array(
    all_entropy
)

predictions = np.array(
    all_predictions
)

labels = np.array(
    all_labels
)


# =========================================================
# SUMMARY
# =========================================================

correct = (
    predictions == labels
)

print("\n========== MC-DROPOUT RESULTS ==========")

print(
    "Mean uncertainty:",
    variance.mean()
)

print(
    "Mean entropy:",
    entropy.mean()
)

print(
    "Correct predictions:",
    correct.sum(),
    "/",
    len(correct)
)

print(
    "Mean uncertainty - correct:",
    variance[correct].mean()
)

if (~correct).sum() > 0:

    print(
        "Mean uncertainty - incorrect:",
        variance[~correct].mean()
    )


# =========================================================
# SAVE PER-IMAGE UNCERTAINTY
# =========================================================

results = pd.DataFrame({

    "filename": all_filenames,

    "true_label": [
        dataset.labels[x]
        for x in labels
    ],

    "predicted_label": [
        dataset.labels[x]
        for x in predictions
    ],

    "uncertainty_variance":
        variance,

    "predictive_entropy":
        entropy,

    "correct":
        correct

})


results.to_csv(
    "mc_dropout_uncertainty.csv",
    index=False
)


# =========================================================
# SAVE FUNCTION
# =========================================================

def mc_dropout_predict(
    image_tensor,
    passes=10
):

    model.eval()

    predictions = []

    with torch.no_grad():

        for _ in range(passes):

            logits = get_logits_with_dropout(
                image_tensor.to(device)
            )

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            predictions.append(
                probabilities
            )

    predictions = torch.stack(
        predictions
    )

    mean_probability = predictions.mean(
        dim=0
    )

    uncertainty = predictions.var(
        dim=0
    ).mean(
        dim=1
    )

    entropy = -(
        mean_probability *
        torch.log(
            mean_probability + 1e-8
        )
    ).sum(
        dim=1
    )

    return (
        mean_probability,
        uncertainty,
        entropy
    )


# =========================================================
# SAVE SUMMARY
# =========================================================

summary = {

    "checkpoint_epoch":
        checkpoint.get("epoch"),

    "mc_passes":
        MC_PASSES,

    "dropout_rate":
        DROPOUT_RATE,

    "validation_images":
        len(dataset),

    "mean_variance":
        float(variance.mean()),

    "mean_entropy":
        float(entropy.mean()),

    "mean_variance_correct":
        float(variance[correct].mean()),

    "mean_variance_incorrect":
        float(
            variance[~correct].mean()
        ) if (~correct).sum() > 0
        else None
}


with open(
    "mc_dropout_summary.json",
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


print("\n================================")
print("Saved: mc_dropout_uncertainty.csv")
print("Saved: mc_dropout_summary.json")
print("================================")