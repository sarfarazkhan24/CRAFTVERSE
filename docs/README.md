# Abstainity

**Abstainity** is a model-agnostic reliability layer for PyTorch image classifiers. It turns overconfident black-box models into calibrated, uncertainty-aware systems that know when to say "I don't know."

## 🚀 The Hackathon Pitch
Standard AI classifiers give you a prediction and a confidence score. But "confidence" is often poorly calibrated, and standard models cannot distinguish between a hard image (ambiguous) and an alien image (out of distribution).

Abstainity fixes this without retraining your model. Just provide an adapter script, and Abstainity wraps your model in three safety layers:
1. **Temperature Scaling**: Calibrates probabilities to reflect true likelihood.
2. **MC-Dropout / TTA**: Computes epistemic uncertainty to detect ambiguous inputs.
3. **Mahalanobis Distance**: Computes OOD distance to reject alien/corrupted inputs.

The result is a guaranteed triage verdict: `ACCEPT`, `UNCERTAIN`, or `ABSTAIN`.

## 📦 Architecture

- **`backend/abstain/`**: The core Python package.
  - `layers/calibration.py`: Temperature scaling wrapper.
  - `layers/mc_dropout.py`: Dropout-based predictive variance estimation (with TTA fallback).
  - `layers/ood.py`: Deep feature Mahalanobis distance calculator.
  - `pipeline.py`: Coordinates the layers and performs threshold-based triage.
  - `fit.py`: Automated pipeline to learn Temperature, Centroids, and Risk Percentiles from a reference set.
- **`backend/api.py`**: A secure FastAPI server that runs user code inside a Sandboxed Subprocess (`worker.py`).
- **`src/`**: A React + Vite frontend that visualizes the AI's internal decision process.

## 🛠️ Capability Tiers (Graceful Degradation)
Abstainity automatically adapts to the capabilities of the provided model:
- **Tier 3 (White-Box + Reference Data)**: Full stack. Temperature scaled, true MC-Dropout variance, and Mahalanobis OOD detection.
- **Tier 2 (White-Box)**: Model has dropout layers but no reference data provided. Outputs raw confidence + true MC-Dropout variance.
- **Tier 1 (Black-Box)**: Model has no dropout and no reference data. Outputs raw confidence + TTA (Test-Time Augmentation) variance + Predictive Entropy.

## 📝 The Adapter Contract
To use Abstainity, provide a simple Python script (`adapter.py`) exposing:

```python
# Required
def load_model():
    return my_pytorch_model

def preprocess(path):
    return my_transforms(image_at_path).unsqueeze(0)

def predict(model, x):
    return model(x) # MUST return raw logits (pre-softmax)

# Optional (Enables Tier 3 OOD Detection)
FEATURE_LAYER = "name_of_penultimate_layer"
```

## 🏃 Quick Start

### 1. Run the Backend API
```bash
cd backend
python -m venv venv
# activate venv
pip install fastapi uvicorn torch torchvision numpy scikit-learn
uvicorn api:app --reload --port 8000
```

### 2. Run the Frontend
```bash
npm install
npm run dev
```
Navigate to `http://localhost:5173`. Upload your `adapter.py` script. Optionally, upload a `reference.zip` containing representative labeled images from your normal domain to enable Tier 3 (OOD detection and calibration), then hit "Run Fitting Pipeline". Finally, upload any test images to see Abstainity in action!
