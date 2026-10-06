import os
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from .layers.ood import MahalanobisOOD
from .adapter import ensure_batched_scores

def calculate_temperature(logits, labels, device="cpu"):
    """Fits temperature using LBFGS."""
    if logits.min() >= 0.0 and torch.allclose(logits.sum(dim=1), torch.ones(logits.size(0)), atol=1e-2):
        logits = torch.log(logits.clamp(min=1e-12))
        
    temperature = nn.Parameter(torch.ones(1, device=device) * 1.0)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def closure():
        optimizer.zero_grad()
        loss = criterion(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return max(float(temperature.item()), 0.01)

def fit_mahalanobis(features, labels, num_classes):
    """Calculates class means and shared covariance."""
    class_means = {}
    features_by_class = {i: [] for i in range(num_classes)}
    
    for f, l in zip(features, labels):
        features_by_class[int(l)].append(f)
        
    for c in range(num_classes):
        if len(features_by_class[c]) > 0:
            class_means[c] = np.mean(features_by_class[c], axis=0)
        else:
            class_means[c] = np.zeros(features.shape[1])
            
    all_centered = []
    for c in range(num_classes):
        if len(features_by_class[c]) > 0:
            centered = features_by_class[c] - class_means[c]
            all_centered.append(centered)
            
    all_centered = np.concatenate(all_centered, axis=0)
    covariance = np.cov(all_centered, rowvar=False)
    covariance += np.eye(covariance.shape[0]) * 1e-4
    covariance_inv = np.linalg.pinv(covariance)
    
    return class_means, covariance_inv

def percentile_rank(values):
    order = np.argsort(np.argsort(values))
    return order / (len(values) - 1 + 1e-8)

def fit_reference_set(adapter, model, reference_dir, output_config_path):
    device = next(model.parameters()).device
    model.eval()

    classes = sorted([d for d in os.listdir(reference_dir) if os.path.isdir(os.path.join(reference_dir, d))])
    num_cls_count = max(len(classes), 1)
    # Dynamic budget to keep total images ~30-45: fast fit (<20s) with enough samples for calibration
    MAX_PER_CLASS = max(3, min(20, 45 // num_cls_count))
    images = []
    labels = []
    if len(classes) == 0:
        flat_imgs = [
            os.path.join(reference_dir, f)
            for f in os.listdir(reference_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        images = flat_imgs[:(MAX_PER_CLASS * 2)]
        labels = [0] * len(images)
        classes = ["default"]
    else:
        class_to_idx = {c: i for i, c in enumerate(classes)}
        for c in classes:
            class_dir = os.path.join(reference_dir, c)
            c_imgs = [
                os.path.join(class_dir, f)
                for f in os.listdir(class_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            # Take up to 3 images per class
            sampled = c_imgs[:MAX_PER_CLASS]
            for img_p in sampled:
                images.append(img_p)
                labels.append(class_to_idx[c])
                
    if len(images) < 2:
        raise ValueError("Reference directory needs at least 2 images for fitting.")

    # Split into Half A and Half B
    try:
        X_A, X_B, y_A, y_B = train_test_split(images, labels, test_size=0.5, stratify=labels, random_state=42)
    except Exception:
        X_A, X_B, y_A, y_B = train_test_split(images, labels, test_size=0.5, shuffle=True, random_state=42)
    
    feature_layer = getattr(adapter, "FEATURE_LAYER", None)
    
    def process_half(X, y):
        all_logits = []
        all_features = []
        hooked_features = []
        handle = None
        
        if feature_layer:
            def hook(module, inp, out):
                if out.dim() > 2:
                    out = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1))
                    out = torch.flatten(out, 1)
                hooked_features.append(out)
            
            layer = dict([*model.named_modules()]).get(feature_layer)
            if layer:
                try:
                    handle = layer.register_forward_hook(hook)
                except Exception:
                    handle = None
        
        with torch.no_grad():
            for img_path in X:
                hooked_features.clear()
                tensor = adapter.preprocess(img_path).to(device)
                scores = ensure_batched_scores(adapter.predict(model, tensor))
                if scores.min() >= 0.0 and torch.allclose(scores.sum(dim=1), torch.ones(scores.size(0), device=scores.device), atol=1e-2):
                    logits = torch.log(scores.clamp(min=1e-12))
                else:
                    logits = scores
                all_logits.append(logits.cpu())
                
                if handle and len(hooked_features) > 0:
                    all_features.append(hooked_features[-1].cpu().numpy()[0])
                
        if handle:
            handle.remove()
            
        return torch.cat(all_logits), np.array(all_features) if all_features else None, torch.tensor(y)

    print(f"Processing Half A ({len(X_A)} images)...")
    logits_A, features_A, labels_A = process_half(X_A, y_A)
    
    temperature = calculate_temperature(logits_A, labels_A, device="cpu")
    print(f"Fitted Temperature: {temperature:.4f}")
    
    class_means, covariance_inv = None, None
    if features_A is not None and len(features_A) > 0:
        class_means, covariance_inv = fit_mahalanobis(features_A, labels_A.numpy(), len(classes))
        print("Fitted Mahalanobis Centroids & Covariance")
        
    # Save partial config for pipeline Half B
    partial_config = {
        "temperature": temperature,
        "class_means": {k: v.tolist() for k, v in class_means.items()} if class_means else None,
        "covariance_inv": covariance_inv.tolist() if covariance_inv is not None else None
    }
    with open(output_config_path, "w") as f:
        json.dump(partial_config, f)
        
    print(f"Processing Half B ({len(X_B)} images) to compute thresholds...")
    from .pipeline import AbstainityPipeline
    pipeline = AbstainityPipeline(output_config_path)
    
    correct_B = []
    risks_B = []
    
    for img_path, label in zip(X_B, y_B):
        tensor = adapter.preprocess(img_path).to(device)
        # Avoid triage inside predict since we don't have thresholds yet
        res = pipeline.predict(adapter, model, tensor)
        
        conf = res["calibrated_confidence"]
        uncert = res["uncertainty"]
        ood = res["ood_score"] or 0.0
        
        correct_B.append(res["prediction"] == label)
        risks_B.append((1.0 - conf, uncert, ood))

    risks_B = np.array(risks_B)
    
    # Extract reference distributions so we can calculate percentiles at inference
    ref_inv_conf = np.sort(risks_B[:, 0]).tolist()
    ref_uncert = np.sort(risks_B[:, 1]).tolist()
    ref_ood = np.sort(risks_B[:, 2]).tolist()
    
    conf_risk = percentile_rank(risks_B[:, 0])
    uncert_risk = percentile_rank(risks_B[:, 1])
    ood_risk = percentile_rank(risks_B[:, 2]) if np.any(risks_B[:, 2] > 0) else np.zeros_like(conf_risk)
    
    total_risk = (conf_risk + uncert_risk + ood_risk) / 3.0
    correct_B = np.array(correct_B)
    
    # Find thresholds
    order = np.argsort(total_risk)
    sorted_risk = total_risk[order]
    sorted_correct = correct_B[order]
    
    best_accept_thresh = sorted_risk[0]
    best_uncertain_thresh = sorted_risk[0]
    
    n = len(sorted_risk)
    has_valid_labels = np.any(sorted_correct)
    if has_valid_labels:
        for i in range(1, n + 1):
            if 1.0 - sorted_correct[:i].mean() <= 0.05:
                best_accept_thresh = sorted_risk[i-1]
                
        for i in range(1, n + 1):
            if 1.0 - sorted_correct[:i].mean() <= 0.15:
                best_uncertain_thresh = sorted_risk[i-1]
    else:
        # Fallback when reference folder labels don't match model output space (e.g. 1000-class ImageNet with arbitrary folder names)
        # Use calibrated risk distribution quantiles
        best_accept_thresh = float(np.percentile(sorted_risk, 75))
        best_uncertain_thresh = float(np.percentile(sorted_risk, 90))
            
    ood_threshold = np.percentile(risks_B[:, 2], 95) if np.any(risks_B[:, 2] > 0) else None
    
    partial_config.update({
        "ref_inv_conf": ref_inv_conf,
        "ref_uncert": ref_uncert,
        "ref_ood": ref_ood,
        "accept_risk_threshold": float(best_accept_thresh),
        "uncertain_risk_threshold": float(best_uncertain_thresh),
        "ood_threshold": float(ood_threshold) if ood_threshold else None
    })
    
    with open(output_config_path, "w") as f:
        json.dump(partial_config, f, indent=2)
        
    print(f"Fit complete. Config saved to {output_config_path}")
    return partial_config
