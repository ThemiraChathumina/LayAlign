import gdown
import zipfile
import os

# Google Drive file ID from the shared link
file_id = "1oT6sfJHlMid0Nn7nrVQkLYwSwJQ53bir"
url = f"https://drive.google.com/uc?id={file_id}"

# Output ZIP path
output_zip = "test.zip"
extract_dir = "./datas/evaluation/"

# Download the ZIP file
gdown.download(url, output_zip, quiet=False)

# Create target directory if it doesn't exist
os.makedirs(extract_dir, exist_ok=True)

# Extract the ZIP file
with zipfile.ZipFile(output_zip, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print(f"Extraction complete. Files are in '{extract_dir}'")

# -------------------------------------------------------

# Google Drive file ID from the shared link
file_id = "1PQSEKwlkGJTxGNzgds1EmmTXLOAQpZg1"
url = f"https://drive.google.com/uc?id={file_id}"

# Output ZIP path
output_zip = "train.zip"
extract_dir = "./datas/query_translation/"

# Download the ZIP file
gdown.download(url, output_zip, quiet=False)

# Create target directory if it doesn't exist
os.makedirs(extract_dir, exist_ok=True)

# Extract the ZIP file
with zipfile.ZipFile(output_zip, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print(f"Extraction complete. Files are in '{extract_dir}'")
