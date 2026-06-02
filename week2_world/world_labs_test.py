import requests
import os
from dotenv import load_dotenv

load_dotenv()
url = "https://api.worldlabs.ai/marble/v1/worlds:generate"
WORLD_API_KEY= os.getenv("WORLD_LABS_KEY")

payload = {
    "display_name": "Mystical Forest",
    "model": "marble-1.1",
    "world_prompt": {
        "type": "text",
        "text_prompt": "A mystical forest with glowing mushrooms"
    }
}
headers = {
    "WLT-Api-Key": f"{WORLD_API_KEY}",
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

print(response.text)

# 0339707d-3682-4d2d-9631-e1ed04e2abdf