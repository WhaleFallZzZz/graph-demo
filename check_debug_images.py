from PIL import Image
from pathlib import Path
import os

def check_images():
    debug_dir = Path("debug_sent_images")
    for img_path in debug_dir.glob("*.png"):
        try:
            img = Image.open(img_path)
            print(f"{img_path.name}: Format={img.format}, Mode={img.mode}, Size={img.size}")
            # Check if it's all black or white
            extrema = img.getextrema()
            print(f"  Extrema: {extrema}")
        except Exception as e:
            print(f"Error checking {img_path}: {e}")

if __name__ == "__main__":
    check_images()
