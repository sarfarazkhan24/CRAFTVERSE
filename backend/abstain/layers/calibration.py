import torch

class TemperatureScaler:
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
        
    def scale(self, logits: torch.Tensor) -> torch.Tensor:
        """Scales logits by temperature and returns calibrated probabilities."""
        return torch.softmax(logits / self.temperature, dim=1)
