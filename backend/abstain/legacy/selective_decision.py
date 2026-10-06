import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


# =========================================================
# SETTINGS
# =========================================================

CSV_FILE = "split_manifest.csv"
IMAGE_DIR = "images"
MODEL_FILE = "model.pt"
TEMP_FILE = "temperature.json"
MC_FILE = "mc_dropout_uncertainty.csv"
OOD_FILE = "ood_scores.csv"

BATCH_SIZE = 16

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

    def __init__(self, csv_file, image_dir):

        df = pd.read_csv(csv_file)

        self.df = df[
            df["split"] == "val"
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

        image = transforms.functional.pil_to_tensor(
            __import__("PIL").Image.open(path).convert("RGB")
        ).float() / 255.0

        image = transforms.functional.resize(
            image,
            [224, 224]
        )

        image = transforms.functional.normalize(
            image,
            mean,
            std
        )

        label = self.label_to_id[
            row["label"]
        ]

        return image, label


dataset = ChestXrayDataset(
    CSV_FILE,
    IMAGE_DIR
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(
    "Validation images:",
    len(dataset)
)


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
    "Checkpoint epoch:",
    checkpoint.get("epoch")
)


# =========================================================
# TEMPERATURE
# =========================================================

with open(TEMP_FILE, "r") as f:
    temp_data = json.load(f)

T = float(
    temp_data["temperature"]
)

print(
    "Temperature:",
    T
)


# =========================================================
# CALIBRATED CONFIDENCE
# =========================================================

all_confidence = []
all_predictions = []
all_labels = []


with torch.no_grad():

    for images, labels in loader:

        images = images.to(device)

        logits = model(images)

        calibrated_logits = (
            logits / T
        )

        probabilities = torch.softmax(
            calibrated_logits,
            dim=1
        )

        confidence, predictions = (
            probabilities.max(dim=1)
        )

        all_confidence.extend(
            confidence.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )


all_confidence = np.array(
    all_confidence
)

all_predictions = np.array(
    all_predictions
)

all_labels = np.array(
    all_labels
)


# =========================================================
# LOAD MC UNCERTAINTY
# =========================================================

mc = pd.read_csv(
    MC_FILE
)

uncertainty = mc[
    "uncertainty_variance"
].values


# =========================================================
# LOAD MAHALANOBIS
# =========================================================

ood = pd.read_csv(
    OOD_FILE
)

ood_score = ood[
    "normal_score"
].values


# =========================================================
# ALIGNMENT CHECK
# =========================================================

n = len(dataset)

assert len(all_confidence) == n
assert len(uncertainty) == n
assert len(ood_score) == n

print(
    "\nAll signals aligned:",
    n
)


# =========================================================
# CONVERT EACH SIGNAL TO RISK
# =========================================================

def percentile_rank(values):

    order = np.argsort(
        np.argsort(values)
    )

    return order / (
        len(values) - 1
    )


# Low confidence = high risk

confidence_risk = percentile_rank(
    1.0 - all_confidence
)

# High uncertainty = high risk

uncertainty_risk = percentile_rank(
    uncertainty
)

# High Mahalanobis = high risk

ood_risk = percentile_rank(
    ood_score
)


# =========================================================
# COMPOSITE RISK
# =========================================================

risk_score = (
    confidence_risk
    + uncertainty_risk
    + ood_risk
) / 3.0


correct = (
    all_predictions ==
    all_labels
)


# =========================================================
# SORT FROM LOWEST TO HIGHEST RISK
# =========================================================

order = np.argsort(
    risk_score
)

sorted_risk = risk_score[
    order
]

sorted_correct = correct[
    order
]


# =========================================================
# FIND ACCEPT THRESHOLD
# =========================================================
#
# Goal:
# Maximum coverage while ACCEPT tier
# has error <= 5%.
#
# =========================================================

best_accept_threshold = None
best_accept_coverage = 0
best_accept_error = None

for i in range(1, n + 1):

    selected_correct = (
        sorted_correct[:i]
    )

    error = (
        1.0 -
        selected_correct.mean()
    )

    coverage = i / n

    if error <= 0.05:

        if coverage > best_accept_coverage:

            best_accept_coverage = coverage

            best_accept_threshold = (
                sorted_risk[i - 1]
            )

            best_accept_error = error


# =========================================================
# FIND UNCERTAIN UPPER THRESHOLD
# =========================================================
#
# UNCERTAIN includes images AFTER ACCEPT
# and must have individual error <= 15%.
#
# We search for the largest UNCERTAIN tier
# satisfying that condition.
#
# =========================================================

accept_mask = (
    risk_score <=
    best_accept_threshold
)

remaining_indices = np.where(
    ~accept_mask
)[0]

remaining_risk = risk_score[
    remaining_indices
]

remaining_correct = correct[
    remaining_indices
]

remaining_order = np.argsort(
    remaining_risk
)

remaining_risk = remaining_risk[
    remaining_order
]

remaining_correct = remaining_correct[
    remaining_order
]


best_uncertain_threshold = None
best_uncertain_count = 0
best_uncertain_error = None

for i in range(
    1,
    len(remaining_risk) + 1
):

    tier_correct = (
        remaining_correct[:i]
    )

    error = (
        1.0 -
        tier_correct.mean()
    )

    if error <= 0.15:

        if i > best_uncertain_count:

            best_uncertain_count = i

            best_uncertain_threshold = (
                remaining_risk[i - 1]
            )

            best_uncertain_error = error


# =========================================================
# FALLBACK
# =========================================================

if best_accept_threshold is None:

    best_accept_threshold = (
        sorted_risk[0]
    )

    best_accept_error = (
        1.0 -
        sorted_correct[0]
    )

    best_accept_coverage = (
        1 / n
    )


if best_uncertain_threshold is None:

    best_uncertain_threshold = (
        best_accept_threshold
    )

    best_uncertain_error = None


# =========================================================
# THREE-TIER DECISION
# =========================================================

def get_decision(risk):

    if risk <= best_accept_threshold:

        return "ACCEPT"

    elif risk <= best_uncertain_threshold:

        return "UNCERTAIN"

    else:

        return "ABSTAIN"


decisions = np.array([
    get_decision(r)
    for r in risk_score
])


# =========================================================
# RESULTS TABLE
# =========================================================

results = pd.DataFrame({

    "filename":
        dataset.df["filename"].values,

    "true_label":
        dataset.df["label"].values,

    "predicted_label":
        [
            dataset.labels[i]
            for i in all_predictions
        ],

    "calibrated_confidence":
        all_confidence,

    "mc_uncertainty":
        uncertainty,

    "mahalanobis_score":
        ood_score,

    "risk_score":
        risk_score,

    "correct":
        correct,

    "decision":
        decisions
})


# =========================================================
# PRINT RESULTS
# =========================================================

print(
    "\n========== STAGE 10 RESULTS =========="
)

print(
    "\nThresholds:"
)

print(
    "ACCEPT threshold:",
    best_accept_threshold
)

print(
    "UNCERTAIN threshold:",
    best_uncertain_threshold
)


for tier in [
    "ACCEPT",
    "UNCERTAIN",
    "ABSTAIN"
]:

    subset = results[
        results["decision"] == tier
    ]

    count = len(subset)

    coverage = count / n

    if count > 0:

        error = (
            1 -
            subset["correct"].mean()
        )

    else:

        error = None

    print(
        f"{tier} | "
        f"images: {count} | "
        f"coverage: {coverage:.4f} | "
        f"error: {error}"
    )


# =========================================================
# SAVE
# =========================================================

results.to_csv(
    "selective_decisions.csv",
    index=False
)


thresholds = {

    "accept_threshold":
        float(best_accept_threshold),

    "uncertain_threshold":
        float(best_uncertain_threshold),

    "accept_target_error":
        0.05,

    "uncertain_target_error":
        0.15,

    "accept_actual_error":
        float(best_accept_error),

    "accept_coverage":
        float(best_accept_coverage),

    "uncertain_actual_error":
        None if best_uncertain_error is None
        else float(best_uncertain_error),

    "temperature":
        float(T),

    "validation_images":
        int(n)
}


with open(
    "decision_thresholds.json",
    "w"
) as f:

    json.dump(
        thresholds,
        f,
        indent=4
    )


# =========================================================
# RISK-COVERAGE CURVE DATA
# =========================================================

coverage = (
    np.arange(1, n + 1) / n
)

cumulative_error = (
    np.cumsum(~sorted_correct)
    /
    np.arange(1, n + 1)
)

risk_coverage = pd.DataFrame({

    "coverage":
        coverage,

    "risk":
        cumulative_error,

    "threshold":
        sorted_risk

})

risk_coverage.to_csv(
    "risk_coverage.csv",
    index=False
)


print(
    "\n======================================"
)

print(
    "Saved: selective_decisions.csv"
)

print(
    "Saved: decision_thresholds.json"
)

print(
    "Saved: risk_coverage.csv"
)

print(
    "======================================"
)