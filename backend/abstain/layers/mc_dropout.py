import torch
import torch.nn as nn

class MCDropout:
    def __init__(self, passes: int = 10):
        self.passes = passes

    def _enable_dropout(self, m: nn.Module):
        """Forces dropout layers to train mode."""
        if type(m) == nn.Dropout or type(m) == nn.Dropout2d or type(m) == nn.Dropout3d:
            m.train()
            
    def _has_dropout(self, m: nn.Module) -> bool:
        """Checks if model has dropout layers."""
        for module in m.modules():
            if type(module) == nn.Dropout or type(module) == nn.Dropout2d or type(module) == nn.Dropout3d:
                return True
        return False

    def get_uncertainty(self, model: nn.Module, x: torch.Tensor, original_predict_fn):
        """
        Runs multiple passes. If dropout exists, uses MC-Dropout (Tier 2).
        If no dropout exists, fallback to standard inference and use entropy (Tier 1).
        
        Returns:
            mean_probs: Averaged probabilities
            uncertainty: The variance (MC-Dropout/TTA) or entropy (fallback)
            tier: 2 for MC-Dropout, 1 for Fallback
            layers: list of string names for layers run
        """
        if not self._has_dropout(model):
            # Tier 1 fallback: No dropout. TTA with Gaussian noise
            model.eval()
            preds = []
            with torch.no_grad():
                for _ in range(self.passes):
                    noise = torch.randn_like(x) * 0.05
                    # Get probabilities directly if predict returns them,
                    # assuming original_predict_fn returns softmax probabilities
                    preds.append(original_predict_fn(model, x + noise))
            
            preds = torch.stack(preds)
            mean_probs = preds.mean(dim=0)
            
            # Check if variance is meaningful (if noise had an effect)
            variance = preds.var(dim=0).mean(dim=1).item()
            
            if variance < 1e-6:
                # Total fallback to entropy
                entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum(dim=1).item()
                return mean_probs, entropy, 1, ["entropy_fallback"]
                
            return mean_probs, variance, 1, ["tta_variance"]

        # Tier 2: MC Dropout
        model.eval()
        model.apply(self._enable_dropout)
        
        preds = []
        with torch.no_grad():
            for _ in range(self.passes):
                preds.append(original_predict_fn(model, x))
                
        preds = torch.stack(preds) # shape: (passes, batch, classes)
        mean_probs = preds.mean(dim=0)
        variance = preds.var(dim=0).mean(dim=1).item() # Mean variance across classes
        
        return mean_probs, variance, 2, ["mc_dropout"]
