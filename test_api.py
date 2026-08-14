import os
from dotenv import load_dotenv
import cohere

load_dotenv()

api_key = os.getenv("COHERE_API_KEY")
print(f"✓ API Key loaded: {api_key[:20]}...")

client = cohere.ClientV2(api_key=api_key)
print("✓ Cohere ClientV2 initialized")

# Test different models
models = ["command-r", "command", "command-r-v1:0"]

for model in models:
    try:
        print(f"\nTesting model: {model}")
        response = client.chat(
            model=model,
            messages=[
                {"role": "user", "content": "What is a startup? Answer in one sentence."}
            ]
        )
        
        print(f"  ✓ API call successful!")
        print(f"  Response type: {type(response)}")
        
        if hasattr(response, 'message') and response.message:
            print(f"  Message content[0].text: {response.message.content[0].text[:100]}...")
        break
        
    except Exception as e:
        error_msg = str(e)
        if "was removed" in error_msg or "not found" in error_msg.lower():
            print(f"  ✗ Model not available")
        else:
            print(f"  ✗ Error: {error_msg[:80]}...")
