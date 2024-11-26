import requests
import re
from urllib.parse import urlparse
import os
import zipfile
from pathlib import Path

def download_image(url, folder):
    try:
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)

        if parsed_url.query:
            filename += re.sub(r"[^\w\-]", "_", parsed_url.query)

        if not filename or filename == "/":
            filename = "downloaded_image"

        if not re.match(r"^.*\.(jpg|jpeg|png|gif)$", filename):
            filename += '.jpg'

        filepath = os.path.join(folder, filename)

        img_file = Path(filepath)
        if img_file.exists():
            return
        
        response = requests.get(url, stream=True)
        if response.status_code == 200:

            os.makedirs(folder, exist_ok=True)

            with open(filepath, "wb") as f:
                f.write(response.content)

            # print(f"Downloaded {filename} from {url}")
    except Exception as e:
        # print(f"Failed to download {url}: {e}")
        return


def zip_selected_folders(output_filename, folder_list, base_directory='.'):
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for folder_name in folder_list:
            folder_path = os.path.join(base_directory, folder_name)
            if os.path.isdir(folder_path):
                for root, dirs, files in os.walk(folder_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, base_directory)
                        zipf.write(file_path, arcname)
