import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY is not set in the environment variables.")

genai.configure(api_key=api_key)

print("--- 사용 가능한 모델 목록 ---")
for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(m.name)
print("--------------------------")
