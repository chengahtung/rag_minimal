import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Load API key from either GOOGLE_GENAI_API_KEY or GENAI_API_KEY
api_key = os.getenv("GOOGLE_GENAI_API_KEY", "") or os.getenv("GENAI_API_KEY", "")
if not api_key:
    raise ValueError("❌ No API key found. Please set GOOGLE_GENAI_API_KEY or GENAI_API_KEY in your .env file.")

# Configure Gemini
genai.configure(api_key=api_key)

# Initialize model
model = genai.GenerativeModel("gemini-2.5-flash")

print("💬 Gemini Chat (type 'exit' to quit)\n")

# Simple chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break

    try:
        # Generate model response
        client = genai.Client()
        model_info = client.models.get(model="gemini-2.5-flash")
        response = model.generate_content(user_input)
        print("Gemini:", response.text.strip(), "\n")
    except Exception as e:
        print(f"⚠️ Error: {e}\n")
