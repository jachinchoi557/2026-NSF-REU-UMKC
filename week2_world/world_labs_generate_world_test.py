import requests
import os
from dotenv import load_dotenv

load_dotenv()

WORLD_KEY = os.getenv("WORLD_LABS_KEY")

url = "https://api.worldlabs.ai/marble/v1/worlds:generate"

payload = {
    "world_prompt": {
        "disable_recaption": True,
        "text_prompt": "Underwater City full of coral reefs.",
        "type": "text"
    },
    "display_name": "water_world_test",
    "model": "marble-1.0",
    "permission": {
        "allow_id_access": False,
        "allowed_readers": [],
        "allowed_writers": [],
        "public": False
    },
    "seed": 2147483647,
    "tags": ["[]"]
}
headers = {
    "WLT-Api-Key": f"{WORLD_KEY}",
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

print(response.text)

# 977242e0-030f-4cd4-9548-03c2cb469027

"""
curl -X GET 'https://api.worldlabs.ai/marble/v1/operations/977242e0-030f-4cd4-9548-03c2cb469027' \
  -H 'WLT-Api-Key: wOCJNtjzU3aNKDKdmyROpBkm5jc5uvEh'
"""