from google import genai
import os
from dotenv import load_dotenv

# YOUR KEY
#API_KEY = "AIzaSyDrcndeFrz2Wh6_eR-z9NG2HBJvjpkVZfk" 

try:
    load_dotenv()
    API_KEY = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=API_KEY)
    print("✅ Authentication Successful. Listing ALL models:")
    print("-" * 40)
    
    # List models - In v1.0, we iterate directly
    for m in client.models.list():
        # In the new SDK, we just print the name directly to see what's available
        print(f"   - {m.name}")
        
    print("-" * 40)
    print("👉 ACTION: Copy one of the names above (e.g., 'gemini-1.5-flash')")
    print("   and paste it into your 'reasoning_agent.py' file.")
        
except Exception as e:
    print(f"❌ Error: {e}")