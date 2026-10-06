import torch
from abstain.pipeline import AbstainityPipeline
from abstain.adapter import load_adapter

# 1. Load the adapter
adapter = load_adapter("example_adapter.py")
model = adapter.load_model()
x = adapter.preprocess("dummy.jpg")

# 2. Run Tier 1/2 prediction (no config file yet, so no T-scaling/OOD)
pipeline = AbstainityPipeline() 
result = pipeline.predict(adapter, model, x)

print("\n--- Pipeline Result (Tier 1/2) ---")
import json
print(json.dumps(result, indent=2))
