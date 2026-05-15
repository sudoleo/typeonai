import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.environ.get("DEVELOPER_GEMINI_API_KEY"))
for m in genai.list_models():
    if "gemini" in m.name:
        print(m.name)
