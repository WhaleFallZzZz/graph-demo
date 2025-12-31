import os
from pathlib import Path
from pypdf import PdfReader
from PIL import Image
import io

def debug_pdf_images(pdf_path):
    print(f"Processing {pdf_path}...")
    reader = PdfReader(pdf_path)
    output_dir = Path("debug_images")
    output_dir.mkdir(exist_ok=True)
    
    # Clean up previous debug images
    for f in output_dir.glob("*"):
        f.unlink()
        
    for i in range(min(5, len(reader.pages))):
        page = reader.pages[i]
        images = page.images
        print(f"Page {i+1} has {len(images)} images.")
        
        for j, image_file in enumerate(images):
            image_bytes = image_file.data
            image_name = image_file.name
            print(f"  Image {j+1}: {image_name}, size={len(image_bytes)} bytes")
            
            try:
                img = Image.open(io.BytesIO(image_bytes))
                print(f"    Format: {img.format}, Size: {img.size}, Mode: {img.mode}")
                
                # Save it
                save_path = output_dir / f"page_{i+1}_img_{j+1}_{image_name}"
                with open(save_path, "wb") as f:
                    f.write(image_bytes)
                print(f"    Saved to {save_path}")
            except Exception as e:
                print(f"    Error opening/saving image: {e}")

if __name__ == "__main__":
    debug_pdf_images("验光配镜100问.pdf")
