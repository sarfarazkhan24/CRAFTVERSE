import os
import sys
sys.path.insert(0, ".")
from abstain.llm_adapter import generate_adapter

try:
    script_text = "print('hello')"
    result = generate_adapter(script_text)
    print("Result:", result)
except Exception as e:
    print("Error:", repr(e))
