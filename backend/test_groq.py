import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path)

api_key = os.environ.get("GROQ_API_KEY")
model_name = os.environ.get("GROQ_MODEL")

if not api_key or not model_name:
    print("GROQ_API_KEY or GROQ_MODEL missing in .env")
else:
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Hello!"}],
            temperature=0
        )
        print("Reply from Groq:", completion.choices[0].message.content)
    except Exception as e:
        print("Error calling Groq:", e)
