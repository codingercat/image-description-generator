import os

# Check if the OPENAI_API_KEY is set
api_key = os.environ.get("OPENAI_API_KEY")
if api_key:
    # Print only the first few characters for security
    masked_key = api_key[:5] + "*" * (len(api_key) - 9) + api_key[-4:]
    print(f"API key is set: {masked_key}")
else:
    print("API key is NOT set in environment variables")

# List all relevant environment variables (without showing sensitive values)
print("\nRelevant environment variables:")
for key in os.environ:
    if key.lower().startswith(('openai', 'api', 'proxy', 'http_proxy', 'https_proxy')):
        value = os.environ[key]
        if 'key' in key.lower() or 'token' in key.lower() or 'secret' in key.lower():
            # Mask sensitive values
            masked_value = value[:4] + "*" * (len(value) - 8) + value[-4:] if len(value) > 8 else "****"
            print(f"{key} = {masked_value}")
        else:
            print(f"{key} = {value}")