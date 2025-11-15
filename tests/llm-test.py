import google.generativeai as genai
from dotenv import load_dotenv
import os

# https://ai.google.dev/gemini-api/docs/api-key
# https://ai.google.dev/gemini-api/docs/rate-limits
# https://ai.google.dev/gemini-api/docs/pricing
# https://ai.google.dev/gemini-api/docs/models


load_dotenv()  # this loads .env into os.environ
api_key = os.getenv("GOOGLE_GENAI_API_KEY", "") or os.getenv("GENAI_API_KEY", "")

# Configure the API key using the correct function
genai.configure(api_key=api_key)

# Now, get an instance of the model
model = genai.GenerativeModel("gemini-2.5-flash")

# And you can use the model to generate content
response = model.generate_content("Hi")


print(response.text)