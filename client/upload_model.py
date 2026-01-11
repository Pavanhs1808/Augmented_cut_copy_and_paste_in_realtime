import os
import requests
from pathlib import Path

# Use environment variable for API URL
API_URL = os.getenv('AR_API_URL', 'http://localhost:5000')

def upload_model(model_path):
    with open(model_path, 'rb') as f:
        files = {'model': f}
        data = {'forward': 'true'}
        try:
            r = requests.post(f"{API_URL}/upload_model", 
                            files=files,
                            data=data,
                            timeout=120)
            return r.json()
        except Exception as e:
            print(f"Upload failed: {e}")
            return None

model_path = r'C:\path\to\your\model.glb'
result = upload_model(model_path)
print(result)