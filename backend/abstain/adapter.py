import importlib.util
import inspect
import sys
import os

import torch


def ensure_batched_scores(scores: torch.Tensor) -> torch.Tensor:
    """Normalize a classifier output to the [batch, classes] shape."""
    if not isinstance(scores, torch.Tensor):
        raise TypeError("Adapter predict() must return a torch.Tensor.")
    if scores.ndim == 1:
        return scores.unsqueeze(0)
    if scores.ndim == 2:
        return scores
    raise ValueError(
        f"Adapter predict() must return a 1D or 2D tensor, got shape {tuple(scores.shape)}."
    )


def load_adapter(script_path: str):
    """
    Loads a user-provided Python script and validates that it meets the Abstainity Adapter Contract.
    
    The contract requires the following functions to be defined:
      - load_model() -> model
      - preprocess(path) -> tensor
      - predict(model, x) -> logits (raw scores before softmax)
    
    Optional constants:
      - FEATURE_LAYER (str)
    """
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Adapter script not found at {script_path}")

    spec = importlib.util.spec_from_file_location("user_adapter", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load adapter script from {script_path}")
        
    user_module = importlib.util.module_from_spec(spec)
    sys.modules["user_adapter"] = user_module
    
    try:
        spec.loader.exec_module(user_module)
    except Exception as e:
        raise RuntimeError(f"Error executing adapter script: {e}")

    # Validate Contract
    required_functions = {
        "load_model": 0,
        "preprocess": 1, 
        "predict": 2, 
    }

    missing_functions = []
    for func_name in required_functions.keys():
        if not hasattr(user_module, func_name):
            missing_functions.append(func_name)
        elif not callable(getattr(user_module, func_name)):
            missing_functions.append(f"{func_name} (must be a callable function)")

    if missing_functions:
        raise ValueError(
            f"Adapter script is missing required contract functions: {', '.join(missing_functions)}. "
            "Please ensure load_model(), preprocess(path), and predict(model, x) are defined."
        )
        
    # Validate optional feature layer
    feature_layer = getattr(user_module, "FEATURE_LAYER", None)
    if feature_layer is not None and not isinstance(feature_layer, str):
         raise ValueError("FEATURE_LAYER must be a string if defined.")
         
    return user_module

def test_adapter(script_path: str):
    """
    Helper function to test if an adapter script is valid and print the result.
    """
    try:
        adapter = load_adapter(script_path)
        print("✅ Adapter loaded successfully and passed contract validation.")
        if hasattr(adapter, "FEATURE_LAYER"):
            print(f"✅ Optional FEATURE_LAYER found: '{adapter.FEATURE_LAYER}'")
        else:
            print("ℹ️ No optional FEATURE_LAYER defined (Mahalanobis OOD may be limited to penultimate outputs or disabled).")
        return adapter
    except Exception as e:
        print(f"❌ Adapter validation failed: {e}")
        return None
