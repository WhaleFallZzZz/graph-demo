import os
import requests
import logging
from pathlib import Path
from llama.ocr_parser import DeepSeekOCRParser
from llama.config import setup_logging

# 设置日志
setup_logging()
logger = logging.getLogger(__name__)

def download_file(url, filename):
    if os.path.exists(filename):
        logger.info(f"File {filename} already exists. Skipping download.")
        return
        
    logger.info(f"Downloading {url} to {filename}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    logger.info("Download complete.")

def main():
    # # 文件 URL (使用之前上下文中的 URL)
    # file_url = "https://public-1316453474.cos.ap-shanghai.myqcloud.com//upload/neo4j/2025-12-30/8dece0c263e6d544_pdf"
    
    # 本地文件名 (模拟用户的文件名，但使用 .pdf 后缀以确保兼容性)
    local_filename = "验光配镜100问.pdf"
    
    # # 1. 下载文件
    # try:
    #     download_file(file_url, local_filename)
    # except Exception as e:
    #     logger.error(f"Failed to download file: {e}")
    #     return

    # 2. 初始化 OCR 解析器 (限制前 5 页)
    logger.info("Initializing DeepSeekOCRParser...")
    parser = DeepSeekOCRParser(max_pages=10)
    
    # 3. 解析文件
    logger.info("Starting OCR processing...")
    try:
        documents = parser.load_data(Path(local_filename))
        
        if documents:
            print("\n" + "="*100)
            print("OCR Result (First 5 Pages):")
            print("="*100 + "\n")
            print(documents[0].text)
            print("\n" + "="*100)
        else:
            print("No text extracted.")
            
    except Exception as e:
        logger.error(f"OCR processing failed: {e}")

if __name__ == "__main__":
    main()
