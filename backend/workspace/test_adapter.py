
import sys
sys.path.insert(0, '.')
import adapter
import torch

model = adapter.load_model()
try:
    dev = next(model.parameters()).device
except Exception:
    dev = torch.device('cpu')

x = torch.randn(1, 3, 224, 224, device=dev)
out = adapter.predict(model, x)
print('SUCCESS_MARKER')
