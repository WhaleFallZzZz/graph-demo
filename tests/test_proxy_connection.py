
import httpx
import os
import sys

def test_connection():
    url = "https://api.siliconflow.cn/v1/models" # Simple endpoint
    print(f"Testing connection to {url}...")
    
    try:
        # We don't need API key just to check TCP connection / TLS handshake usually, 
        # but let's see if we get a response (even 401 is fine, means connection worked).
        response = httpx.get(url, timeout=10)
        print(f"Status Code: {response.status_code}")
        print("✅ Connection successful!")
        return True
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        # Check if it's related to socks
        if "socks" in str(e).lower():
            print("❗ This error is related to SOCKS proxy support.")
        return False

if __name__ == "__main__":
    if test_connection():
        sys.exit(0)
    else:
        sys.exit(1)
