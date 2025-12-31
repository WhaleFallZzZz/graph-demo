import fitz  # PyMuPDF

def debug_fitz(pdf_path):
    print(f"Processing {pdf_path} with PyMuPDF...")
    doc = fitz.open(pdf_path)
    
    # Process first 5 pages
    for i in range(min(5, len(doc))):
        page = doc.load_page(i)
        pix = page.get_pixmap(dpi=150) # 150 DPI is usually enough for OCR
        output_path = f"debug_images/fitz_page_{i+1}.png"
        pix.save(output_path)
        print(f"Saved {output_path} ({pix.width}x{pix.height})")

if __name__ == "__main__":
    debug_fitz("验光配镜100问.pdf")
