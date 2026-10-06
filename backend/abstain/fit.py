import os
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from .layers.ood import MahalanobisOOD

def calculate_temperature(logits, labels, device="cpu"):
    """Fits temperature using LBFGS."""
    temperature = nn.Parameter(torch.ones(1, device=device) * 1.0)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def closure():
        optimizer.zero_grad()
        loss = criterion(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return temperature.item()

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
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    images = []
    labels = []
    for c in classes:
        class_dir = os.path.join(reference_dir, c)
        for f in os.listdir(class_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                images.append(os.path.join(class_dir, f))
                labels.append(class_to_idx[c])
                
    if len(images) < 10:
        raise ValueError("Reference directory needs at least 10 images for fitting.")

    # Split into Half A and Half B
    X_A, X_B, y_A, y_B = train_test_split(images, labels, test_size=0.5, stratify=labels, random_state=42)
    
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
                handle = layer.register_forward_hook(hook)
        
        with torch.no_grad():
            for img_path in X:
                tensor = adapter.preprocess(img_path).to(device)
                logits = adapter.predict(model, tensor)
                all_logits.append(logits.cpu())
                
                if handle and len(hooked_features) > 0:
                    all_features.append(hooked_features.pop().cpu().numpy()[0])
                
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
    for i in range(1, n + 1):
        if 1.0 - sorted_correct[:i].mean() <= 0.05:
            best_accept_thresh = sorted_risk[i-1]
            
    for i in range(1, n + 1):
        if 1.0 - sorted_correct[:i].mean() <= 0.15:
            best_uncertain_thresh = sorted_risk[i-1]
            
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
