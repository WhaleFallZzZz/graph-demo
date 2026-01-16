"""
文件哈希工具模块

提供文件哈希计算功能，用于文件去重和完整性校验。
"""

import os
import hashlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_file_hash(file_path: str, algorithm: str = 'md5') -> str:
    """
    计算文件内容的哈希值
    
    用于文件去重、完整性校验等场景。支持多种哈希算法，
    包括 MD5、SHA1、SHA256 等。
    
    Args:
        file_path: 文件路径
        algorithm: 哈希算法，支持 'md5'、'sha1'、'sha256' 等
        
    Returns:
        十六进制哈希字符串
        
    Raises:
        FileNotFoundError: 文件不存在
        IOError: 文件读取失败
        
    使用示例：
        >>> get_file_hash('test.txt')
        'd41d8cd98f00b204e9800998ecf8427e'
        
        >>> get_file_hash('test.txt', algorithm='sha256')
        'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    hash_func = hashlib.new(algorithm)
    
    try:
        with open(file_path, 'rb') as f:
            # 分块读取以高效处理大文件
            for chunk in iter(lambda: f.read(4096), b''):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
    except IOError as e:
        logger.error(f"Failed to read file {file_path}: {e}")
        raise
