import os
from dotenv import load_dotenv
import cohere

load_dotenv()
api_key = os.getenv("COHERE_API_KEY")

client = cohere.ClientV2(api_key=api_key)

# Try to list models or find available ones
test_models = [
    "command",
    "command-light",
    "command-nightly",
    "command-r",
    "command-r-plus",
    "command-r-v1:0",
    "c4ai-aya-expanse-32b",
    "aya-expanse-8b",
]

print("Testing models:\n")

for model in test_models:
    try:
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": "Say OK"}]
        )
        print(f"✓ {model:<30} WORKS!")
        break
    except cohere.errors.not_found_error.NotFoundError as e:
        if "was removed" in str(e) or "not found" in str(e).lower():
            print(f"✗ {model:<30} Not available")
    except Exception as e:
        error = str(e)[:80]
        print(f"? {model:<30} Error: {error}")
