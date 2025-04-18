import openai

# Azure OpenAI credentials
openai.api_type = "azure"
openai.api_key = "9brWImpH0k4AQWp9jiLAZHKUKPItW1K4x0yvmt4b6HEpf5vRRkPpJQQJ99BDACqBBLyXJ3w3AAABACOGRrE1"
openai.api_base = "https://yashiii.openai.azure.com/"
openai.api_version = "2023-12-01-preview"  # or check your Azure version

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
