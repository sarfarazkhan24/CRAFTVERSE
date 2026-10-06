import torch
import torch.nn as nn
import numpy as np

class MahalanobisOOD:
    def __init__(self, class_means=None, covariance_inv=None):
        """
        class_means: Dict of class index (int) to numpy array (1D)
        covariance_inv: Numpy 2D array of the inverted covariance matrix
        """
        # Ensure correct types if loaded from JSON
        self.class_means = class_means
        if self.class_means is not None and isinstance(list(self.class_means.values())[0], list):
            self.class_means = {int(k): np.array(v) for k, v in self.class_means.items()}
            
        self.covariance_inv = covariance_inv
        if self.covariance_inv is not None and isinstance(self.covariance_inv, list):
            self.covariance_inv = np.array(self.covariance_inv)
        
    def hook_layer(self, model, feature_layer_name):
        """Registers a forward hook to extract features from the specified layer."""
        features = []
        def hook(module, input, output):
            # Flatten spatial dimensions if needed (like AdaptiveAvgPool + Flatten)
            if output.dim() > 2:
                out = torch.nn.functional.adaptive_avg_pool2d(output, (1, 1))
                out = torch.flatten(out, 1)
            else:
                out = output
            features.append(out)
            
        layer = dict([*model.named_modules()]).get(feature_layer_name)
        if layer is None:
            raise ValueError(f"Layer '{feature_layer_name}' not found in model.")
            
        handle = layer.register_forward_hook(hook)
        return handle, features

    def compute_distance(self, feature: np.ndarray) -> float:
        """Computes the minimum Mahalanobis distance to any known class centroid."""
        if self.class_means is None or self.covariance_inv is None:
            return None # Cannot compute if not fitted
            
        scores = []
        for mean in self.class_means.values():
            diff = feature - mean
            # distance = diff^T * Cov^-1 * diff
            dist = np.dot(np.dot(diff, self.covariance_inv), diff.T)
            scores.append(dist)
        return float(min(scores))
