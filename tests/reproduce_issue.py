
import sys
import os
from llama_index.core.schema import Document

try:
    doc = Document(text="Hello world")
    print(f"Available attributes: {dir(doc)}")
    
    # Try alternate ways to set text
    if hasattr(doc, "set_content"):
        print("Trying set_content...")
        doc.set_content("Hello Llama")
        print(f"New text: {doc.text}")
        if doc.text == "Hello Llama":
            print("✅ set_content worked!")
        else:
            print("❌ set_content failed to update doc.text")
    elif hasattr(doc, "text_content"):
         # Some versions might have this
         pass
         
except Exception as e:
    print(f"Error: {e}")
