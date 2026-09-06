from PIL import Image
import os

folder = "data/Toronto_ISPRS/train"
bad_files = []

for filename in os.listdir(folder):
    if filename.endswith(".png"):
        try:
            Image.open(os.path.join(folder, filename)).verify()
        except Exception:
            bad_files.append(filename)

print(f"Found {len(bad_files)} corrupted files out of {len(os.listdir(folder))}")