import os
import requests
from concurrent.futures import ThreadPoolExecutor

letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
base_url = "https://raw.githubusercontent.com/dain-kim/ASLingo/main/guide_videos/{}.mp4"
output_dir = "ASL_LETTERS"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def download_letter(letter):
    url = base_url.format(letter)
    path = os.path.join(output_dir, f"{letter}.mp4")
    print(f"Downloading {letter}...")
    response = requests.get(url, timeout=30)
    if response.status_code == 200:
        with open(path, "wb") as f:
            f.write(response.content)
        print(f"Finished {letter}")
    else:
        print(f"Failed {letter}: {response.status_code}")

with ThreadPoolExecutor(max_workers=5) as executor:
    executor.map(download_letter, letters)

print("All downloads completed.")
