import openai
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Azure OpenAI credentials
openai.api_type = os.getenv("OPENAI_API_TYPE")
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_version = os.getenv("OPENAI_API_VERSION")

# Use ChatCompletion (for GPT-3.5, GPT-4, etc.)
response = openai.ChatCompletion.create(
    engine="gpt35-test",  # not the model name! Use the deployment name
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ],
    max_tokens=100
)

# Output the response
print(response['choices'][0]['message']['content'])