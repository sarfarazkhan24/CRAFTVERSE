# How Abstainity Works

Abstainity is a "reliability layer" that wraps around any PyTorch image classifier. Standard neural networks are notoriously overconfident—they will often output "99% confident" even when looking at completely random noise or an image they've never seen before.

Abstainity solves this by giving your model the ability to say **"I don't know"**.

It does this progressively, using three distinct capability tiers depending on what information you provide:

### Tier 1: The Black-Box Fallback (Entropy & TTA)
If you just upload a model with no extra information, Abstainity uses Tier 1. 
Instead of trusting the highest percentage, it looks at the **Entropy** of the prediction (how "spread out" the confidence is across all possible classes) and uses **Test-Time Augmentation (TTA)**. It subtly modifies the image (e.g., slight crops/flips) and runs it multiple times. If the model's prediction changes wildly on these slight variations, Abstainity flags it as a guess and returns **ABSTAIN/UNCERTAIN**.

### Tier 2: The White-Box Interception (MC Dropout)
If Abstainity detects Dropout layers in your PyTorch model, it automatically promotes the analysis to Tier 2.
Normally, Dropout is only used during training. Abstainity *forces* Dropout to stay on during testing, and runs the image through the model multiple times (e.g., 10-20 passes). Because different neurons are randomly turned off each time, a truly confident model will still make the same prediction. A guessing model will completely change its mind every pass. Abstainity calculates the variance (Epistemic Uncertainty) across these passes. High variance = **ABSTAIN**.

### Tier 3: The Calibrated Reference (Mahalanobis OOD)
If you provide a small "Reference Dataset" (e.g., 50 normal images per class), Abstainity reaches its maximum capability.
1. **Temperature Scaling:** It learns how to accurately scale the model's raw confidence percentages so that a "70% confidence" actually means it is right 70% of the time.
2. **Mahalanobis Out-of-Distribution (OOD):** It intercepts the raw internal features (the `FEATURE_LAYER`) of your model before the final prediction is made. It builds a mathematical map (covariance matrix) of what "normal" features look like. When an alien image (like a hand in an X-ray dataset) is passed through, its internal features will fall far outside this map, and Abstainity immediately flags it as OOD and returns **ABSTAIN**, regardless of what the final prediction percentage is.
