import openai
print(f"OpenAI library version: {openai.__version__}")

# Try both import styles to see which one works
try:
    from openai import OpenAI
    client = OpenAI()
    print("Successfully imported and initialized OpenAI client using 'from openai import OpenAI'")
except Exception as e:
    print(f"Error with 'from openai import OpenAI': {e}")

try:
    client = openai.OpenAI()
    print("Successfully initialized OpenAI client using 'openai.OpenAI()'")
except Exception as e:
    print(f"Error with 'openai.OpenAI()': {e}")