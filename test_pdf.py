
from llama_index.core import SimpleDirectoryReader
import os

# Create a dummy text file to verify basic reading
with open("test.txt", "w") as f:
    f.write("Hello world.")

# Try to read
try:
    reader = SimpleDirectoryReader(input_files=["test.txt"])
    docs = reader.load_data()
    print(f"TXT Read: {docs[0].text}")
except Exception as e:
    print(f"TXT Failed: {e}")

# Check if pypdf is installed
try:
    import pypdf
    print(f"pypdf version: {pypdf.__version__}")
except ImportError:
    print("pypdf not installed")

