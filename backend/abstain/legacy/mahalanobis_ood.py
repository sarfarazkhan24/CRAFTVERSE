import os
import pickle
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import roc_auc_score, roc_curve


# =========================================================
# SETTINGS
# =========================================================

CSV_FILE = "split_manifest.csv"
IMAGE_DIR = "images"
MODEL_FILE = "model.pt"

BATCH_SIZE = 16

NOISE_STD = 0.20

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

    def __init__(
        self,
        csv_file,
        image_dir,
        split
    ):

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
            for i, label in enumerate(
                self.labels
            )
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

        return image, label


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
print("Classes:", dataset.labels)


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
# FEATURE EXTRACTION
# =========================================================

def get_features(images):

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

    return features


# =========================================================
# COLLECT TRAIN FEATURES
# =========================================================
#
# IMPORTANT:
# Mahalanobis reference statistics MUST be learned
# from TRAINING data only.
#
# Validation data is used only for evaluation.
# =========================================================

train_dataset = ChestXrayDataset(
    CSV_FILE,
    IMAGE_DIR,
    "train"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

features_by_class = {
    i: []
    for i in range(len(train_dataset.labels))
}

print("\nCollecting training features...")


with torch.no_grad():

    for batch_idx, (
        images,
        labels
    ) in enumerate(train_loader):

        images = images.to(device)

        features = get_features(
            images
        )

        features = features.cpu().numpy()

        labels = labels.numpy()

        for feature, label in zip(
            features,
            labels
        ):

            features_by_class[
                int(label)
            ].append(feature)

        if (batch_idx + 1) % 50 == 0:

            print(
                "Processed training batches:",
                batch_idx + 1
            )


# =========================================================
# CLASS MEANS
# =========================================================

class_means = {}

for class_id in range(
    len(train_dataset.labels)
):

    class_features = np.array(
        features_by_class[class_id]
    )

    class_means[class_id] = (
        class_features.mean(axis=0)
    )

    print(
        train_dataset.labels[class_id],
        "samples:",
        len(class_features)
    )


# =========================================================
# SHARED COVARIANCE
# =========================================================

print("\nCalculating shared covariance...")

all_centered = []

for class_id in range(
    len(train_dataset.labels)
):

    class_features = np.array(
        features_by_class[class_id]
    )

    centered = (
        class_features -
        class_means[class_id]
    )

    all_centered.append(
        centered
    )

all_centered = np.concatenate(
    all_centered,
    axis=0
)

# Covariance matrix

covariance = np.cov(
    all_centered,
    rowvar=False
)

# Small regularization for numerical stability

covariance += (
    np.eye(covariance.shape[0])
    * 1e-4
)

covariance_inv = np.linalg.pinv(
    covariance
)

print(
    "Feature dimension:",
    all_centered.shape[1]
)


# =========================================================
# MAHALANOBIS FUNCTION
# =========================================================

def mahalanobis_distance(
    features
):

    scores = []

    for class_id in range(
        len(class_means)
    ):

        mean = class_means[class_id]

        diff = (
            features -
            mean
        )

        distance = np.einsum(
            "ij,jk,ik->i",
            diff,
            covariance_inv,
            diff
        )

        scores.append(
            distance
        )

    scores = np.stack(
        scores,
        axis=1
    )

    # Minimum distance to any known class

    return scores.min(
        axis=1
    )


# =========================================================
# NORMAL VALIDATION SCORES
# =========================================================

normal_scores = []

print("\nCalculating normal validation scores...")

with torch.no_grad():

    for images, labels in loader:

        images = images.to(device)

        features = get_features(
            images
        )

        features = features.cpu().numpy()

        scores = mahalanobis_distance(
            features
        )

        normal_scores.extend(
            scores
        )

normal_scores = np.array(
    normal_scores
)


# =========================================================
# CORRUPTED VALIDATION SCORES
# =========================================================

corrupted_scores = []

print("\nCalculating corrupted-image scores...")

with torch.no_grad():

    for images, labels in loader:

        # Add Gaussian noise

        noise = torch.randn_like(
            images
        ) * NOISE_STD

        corrupted = (
            images + noise
        )

        corrupted = torch.clamp(
            corrupted,
            -3,
            3
        )

        corrupted = corrupted.to(
            device
        )

        features = get_features(
            corrupted
        )

        features = features.cpu().numpy()

        scores = mahalanobis_distance(
            features
        )

        corrupted_scores.extend(
            scores
        )

corrupted_scores = np.array(
    corrupted_scores
)


# =========================================================
# OOD EVALUATION
# =========================================================

y_true = np.concatenate([
    np.zeros(
        len(normal_scores)
    ),

    np.ones(
        len(corrupted_scores)
    )
])

y_scores = np.concatenate([
    normal_scores,
    corrupted_scores
])


auroc = roc_auc_score(
    y_true,
    y_scores
)

fpr, tpr, thresholds = roc_curve(
    y_true,
    y_scores
)

# FPR at approximately 95% TPR

idx = np.where(
    tpr >= 0.95
)[0]

if len(idx) > 0:

    fpr95 = fpr[idx[0]]

else:

    fpr95 = None


# =========================================================
# PRINT RESULTS
# =========================================================

print("\n========== MAHALANOBIS OOD RESULTS ==========")

print(
    "Normal mean score:",
    normal_scores.mean()
)

print(
    "Corrupted mean score:",
    corrupted_scores.mean()
)

print(
    "AUROC:",
    auroc
)

print(
    "FPR95:",
    fpr95
)


# =========================================================
# SAVE PARAMETERS
# =========================================================

with open(
    "ood_params.pkl",
    "wb"
) as f:

    pickle.dump(
        {
            "class_means":
                class_means,

            "covariance_inv":
                covariance_inv,

            "classes":
                train_dataset.labels
        },
        f
    )


# =========================================================
# SAVE SCORES
# =========================================================

ood_results = pd.DataFrame({

    "normal_score":
        normal_scores,

    "corrupted_score":
        corrupted_scores

})

ood_results.to_csv(
    "ood_scores.csv",
    index=False
)


# =========================================================
# SAVE SUMMARY
# =========================================================

summary = {

    "checkpoint_epoch":
        checkpoint.get("epoch"),

    "validation_images":
        len(dataset),

    "noise_std":
        NOISE_STD,

    "normal_mean_score":
        float(normal_scores.mean()),

    "corrupted_mean_score":
        float(corrupted_scores.mean()),

    "AUROC":
        float(auroc),

    "FPR95":
        None if fpr95 is None
        else float(fpr95)
}


with open(
    "ood_results.json",
    "w"
) as f:

    import json

    json.dump(
        summary,
        f,
        indent=4
    )


print("\n================================")
print("Saved: ood_params.pkl")
print("Saved: ood_scores.csv")
print("Saved: ood_results.json")
print("================================")