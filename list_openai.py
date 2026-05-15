import os
import openai
from dotenv import load_dotenv

load_dotenv()
client = openai.OpenAI(api_key=os.environ.get("DEVELOPER_OPENAI_API_KEY"))
for m in client.models.list():
    if "gpt" in m.id or "o1" in m.id or "o3" in m.id:
        print(m.id)
