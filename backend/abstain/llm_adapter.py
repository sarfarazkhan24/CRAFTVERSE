import os
import re
from dotenv import load_dotenv

# Load env specifically from backend/.env
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path=env_path)

def generate_adapter(script_text: str, error_hint: str = None) -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    model_name = os.environ.get("GROQ_MODEL")
    
    if not api_key or not model_name:
        print("LLM fallback unavailable")
        return None
        
    try:
        from groq import Groq
        client = Groq(api_key=api_key, timeout=15.0)
        
        system_prompt = (
            "Output ONLY Python code, no markdown fences or explanation. "
            "Define: load_model() -> model in eval mode; "
            "preprocess(path) -> tensor shaped [1,C,H,W]; "
            "predict(model, x) -> 1D probability tensor (apply softmax if the script outputs logits); "
            "FEATURE_LAYER = '<name of last layer before the classifier head>' or None. "
            "Reuse the script's own code and constants. Remove training code, argparse, and top-level execution. "
            "Do not invent file paths."
        )
        
        user_message = f"Here is the script:\n\n{script_text}"
        if error_hint:
            user_message += f"\n\nFix the following error from your previous attempt:\n{error_hint}"
            
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0,
        )
        
        response_text = completion.choices[0].message.content
        # Strip markdown fences
        response_text = re.sub(r"^```python\n", "", response_text.strip())
        response_text = re.sub(r"^```\n", "", response_text)
        response_text = re.sub(r"\n```$", "", response_text)
        
        return response_text
    except Exception as e:
        return None
