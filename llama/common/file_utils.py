"""
知识图谱系统的文件操作工具。

本模块提供统一的文件处理函数，包括：
- 文件类型检测
- 文件哈希（用于去重）
- 内容读取和写入
- 文件验证
- 目录操作
"""

import os
import hashlib
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union

logger = logging.getLogger(__name__)

# 支持的文档处理文件扩展名
SUPPORTED_EXTENSIONS = {
    '.txt': 'text',
    '.md': 'markdown',
    '.pdf': 'pdf',
    '.docx': 'docx',
    '.doc': 'doc',
    '.json': 'json',
    '.csv': 'csv'
}

# 最大文件大小（字节）
MAX_FILE_SIZES = {
    'text': 10 * 1024 * 1024,      # 10 MB
    'pdf': 200 * 1024 * 1024,       # 50 MB
    'docx': 100 * 1024 * 1024,      # 20 MB
    'doc': 100 * 1024 * 1024,       # 20 MB
    # 'json': 5 * 1024 * 1024,       # 5 MB
    # 'csv': 10 * 1024 * 1024        # 10 MB
}


def get_file_hash(file_path: str, algorithm: str = 'md5') -> str:
    """
    Generate hash for file content.
    
    Args:
        file_path: Path to the file
        algorithm: Hash algorithm to use ('md5', 'sha1', 'sha256')
        
    Returns:
        Hexadecimal hash string
        
    Raises:
        FileNotFoundError: If file doesn't exist
        IOError: If file cannot be read
        
    Examples:
        >>> get_file_hash('test.txt')
        'd41d8cd98f00b204e9800998ecf8427e'
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    hash_func = hashlib.new(algorithm)
    
    try:
        with open(file_path, 'rb') as f:
            # Read in chunks to handle large files efficiently
            for chunk in iter(lambda: f.read(4096), b''):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
    except IOError as e:
        logger.error(f"Failed to read file {file_path}: {e}")
        raise


def detect_file_type(file_path: str) -> Optional[str]:
    """
    Detect file type based on extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File type string or None if unsupported
        
    Examples:
        >>> detect_file_type('document.pdf')
        'pdf'
        >>> detect_file_type('data.unknown')
        None
    """
    ext = os.path.splitext(file_path)[1].lower()
    return SUPPORTED_EXTENSIONS.get(ext)


def is_supported_file(file_path: str) -> bool:
    """
    Check if file type is supported.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if supported, False otherwise
        
    Examples:
        >>> is_supported_file('document.pdf')
        True
        >>> is_supported_file('image.jpg')
        False
    """
    return detect_file_type(file_path) is not None


def get_file_size(file_path: str) -> int:
    """
    Get file size in bytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in bytes
        
    Raises:
        FileNotFoundError: If file doesn't exist
        
    Examples:
        >>> get_file_size('test.txt')
        1024
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return os.path.getsize(file_path)


def is_file_size_valid(file_path: str) -> bool:
    """
    Check if file size is within allowed limits.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if size is valid, False otherwise
        
    Examples:
        >>> is_file_size_valid('small_file.txt')
        True
    """
    try:
        file_type = detect_file_type(file_path)
        if not file_type:
            return False
        
        file_size = get_file_size(file_path)
        max_size = MAX_FILE_SIZES.get(file_type, MAX_FILE_SIZES['text'])
        
        return file_size <= max_size
    except Exception as e:
        logger.error(f"Error checking file size: {e}")
        return False


def read_file_content(file_path: str, 
                      encoding: str = 'utf-8',
                      max_size: Optional[int] = None) -> str:
    """
    Read content from a text file.
    
    Args:
        file_path: Path to the file
        encoding: File encoding (default: 'utf-8')
        max_size: Maximum bytes to read (None for no limit)
        
    Returns:
        File content as string
        
    Raises:
        FileNotFoundError: If file doesn't exist
        UnicodeDecodeError: If encoding fails
        
    Examples:
        >>> read_file_content('test.txt')
        'Hello, World!'
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            if max_size:
                content = f.read(max_size)
            else:
                content = f.read()
            
            return content
    except UnicodeDecodeError as e:
        logger.error(f"Encoding error reading {file_path}: {e}")
        raise


def write_file_content(file_path: str, 
                       content: str,
                       encoding: str = 'utf-8',
                       create_dirs: bool = True) -> None:
    """
    Write content to a text file.
    
    Args:
        file_path: Path to the file
        content: Content to write
        encoding: File encoding (default: 'utf-8')
        create_dirs: Create parent directories if they don't exist
        
    Raises:
        IOError: If file cannot be written
        
    Examples:
        >>> write_file_content('output.txt', 'Hello, World!')
    """
    if create_dirs:
        parent_dir = os.path.dirname(file_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
    
    try:
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(content)
        
        logger.info(f"Successfully wrote to {file_path}")
    except IOError as e:
        logger.error(f"Failed to write to {file_path}: {e}")
        raise


def read_binary_file(file_path: str, max_size: Optional[int] = None) -> bytes:
    """
    Read content from a binary file.
    
    Args:
        file_path: Path to the file
        max_size: Maximum bytes to read (None for no limit)
        
    Returns:
        File content as bytes
        
    Raises:
        FileNotFoundError: If file doesn't exist
        
    Examples:
        >>> read_binary_file('image.png')
        b'\\x89PNG...'
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        with open(file_path, 'rb') as f:
            if max_size:
                content = f.read(max_size)
            else:
                content = f.read()
            
            return content
    except IOError as e:
        logger.error(f"Failed to read binary file {file_path}: {e}")
        raise


def write_binary_file(file_path: str, 
                      content: bytes,
                      create_dirs: bool = True) -> None:
    """
    Write content to a binary file.
    
    Args:
        file_path: Path to the file
        content: Content to write
        create_dirs: Create parent directories if they don't exist
        
    Raises:
        IOError: If file cannot be written
        
    Examples:
        >>> write_binary_file('output.bin', b'\\x00\\x01')
    """
    if create_dirs:
        parent_dir = os.path.dirname(file_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
    
    try:
        with open(file_path, 'wb') as f:
            f.write(content)
        
        logger.info(f"Successfully wrote binary data to {file_path}")
    except IOError as e:
        logger.error(f"Failed to write binary file {file_path}: {e}")
        raise


def ensure_directory(directory: str) -> None:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Path to the directory
        
    Examples:
        >>> ensure_directory('/path/to/dir')
    """
    os.makedirs(directory, exist_ok=True)
    logger.debug(f"Ensured directory exists: {directory}")


def list_files(directory: str,
               pattern: Optional[str] = None,
               recursive: bool = False) -> List[str]:
    """
    List files in a directory.
    
    Args:
        directory: Path to the directory
        pattern: Optional glob pattern to filter files
        recursive: Whether to search recursively
        
    Returns:
        List of file paths
        
    Examples:
        >>> list_files('/path/to/dir', '*.txt')
        ['/path/to/dir/file1.txt', '/path/to/dir/file2.txt']
    """
    path = Path(directory)
    
    if not path.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return []
    
    if recursive:
        files = path.rglob('*') if not pattern else path.rglob(pattern)
    else:
        files = path.glob('*') if not pattern else path.glob(pattern)
    
    # Filter to only files (not directories)
    file_paths = [str(f) for f in files if f.is_file()]
    
    return sorted(file_paths)


def get_file_extension(file_path: str) -> str:
    """
    Get file extension without the dot.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File extension (lowercase, without dot)
        
    Examples:
        >>> get_file_extension('document.pdf')
        'pdf'
    """
    return os.path.splitext(file_path)[1][1:].lower()


def get_file_name(file_path: str, with_extension: bool = True) -> str:
    """
    Get file name from path.
    
    Args:
        file_path: Path to the file
        with_extension: Whether to include extension
        
    Returns:
        File name
        
    Examples:
        >>> get_file_name('/path/to/document.pdf')
        'document.pdf'
        >>> get_file_name('/path/to/document.pdf', with_extension=False)
        'document'
    """
    name = os.path.basename(file_path)
    
    if not with_extension:
        name = os.path.splitext(name)[0]
    
    return name


def get_parent_directory(file_path: str) -> str:
    """
    Get parent directory of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Parent directory path
        
    Examples:
        >>> get_parent_directory('/path/to/file.txt')
        '/path/to'
    """
    return os.path.dirname(file_path)


def join_paths(*paths: str) -> str:
    """
    Join multiple path components.
    
    Args:
        *paths: Path components to join
        
    Returns:
        Joined path
        
    Examples:
        >>> join_paths('/path', 'to', 'file.txt')
        '/path/to/file.txt'
    """
    return os.path.join(*paths)


def is_absolute_path(path: str) -> bool:
    """
    Check if path is absolute.
    
    Args:
        path: Path to check
        
    Returns:
        True if absolute, False otherwise
        
    Examples:
        >>> is_absolute_path('/path/to/file')
        True
        >>> is_absolute_path('relative/path')
        False
    """
    return os.path.isabs(path)


def normalize_path(path: str) -> str:
    """
    Normalize a path (resolve . and .., etc.).
    
    Args:
        path: Path to normalize
        
    Returns:
        Normalized path
        
    Examples:
        >>> normalize_path('/path/to/../file')
        '/path/file'
    """
    return os.path.normpath(path)


def file_exists(file_path: str) -> bool:
    """
    Check if a file exists.
    
    Args:
        file_path: Path to check
        
    Returns:
        True if file exists, False otherwise
        
    Examples:
        >>> file_exists('existing_file.txt')
        True
    """
    return os.path.isfile(file_path)


def directory_exists(directory: str) -> bool:
    """
    Check if a directory exists.
    
    Args:
        directory: Path to check
        
    Returns:
        True if directory exists, False otherwise
        
    Examples:
        >>> directory_exists('/path/to/dir')
        True
    """
    return os.path.isdir(directory)


def delete_file(file_path: str) -> bool:
    """
    Delete a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if deleted successfully, False otherwise
        
    Examples:
        >>> delete_file('temp_file.txt')
        True
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Deleted file: {file_path}")
            return True
        return False
    except Exception as e:
        logger.error(f"Failed to delete file {file_path}: {e}")
        return False


def delete_directory(directory: str, recursive: bool = False) -> bool:
    """
    Delete a directory.
    
    Args:
        directory: Path to the directory
        recursive: Whether to delete recursively
        
    Returns:
        True if deleted successfully, False otherwise
        
    Examples:
        >>> delete_directory('temp_dir', recursive=True)
        True
    """
    try:
        if not os.path.exists(directory):
            return False
        
        if recursive:
            import shutil
            shutil.rmtree(directory)
        else:
            os.rmdir(directory)
        
        logger.info(f"Deleted directory: {directory}")
        return True
    except Exception as e:
        logger.error(f"Failed to delete directory {directory}: {e}")
        return False


def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Get comprehensive file information.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary with file information
        
    Examples:
        >>> get_file_info('test.txt')
        {'name': 'test.txt', 'size': 1024, 'type': 'text', ...}
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    stat = os.stat(file_path)
    
    return {
        'name': get_file_name(file_path),
        'path': os.path.abspath(file_path),
        'size': stat.st_size,
        'type': detect_file_type(file_path),
        'extension': get_file_extension(file_path),
        'created': stat.st_ctime,
        'modified': stat.st_mtime,
        'accessed': stat.st_atime,
        'is_file': os.path.isfile(file_path),
        'is_dir': os.path.isdir(file_path)
    }


def copy_file(source: str, destination: str, overwrite: bool = False) -> bool:
    """
    Copy a file from source to destination.
    
    Args:
        source: Source file path
        destination: Destination file path
        overwrite: Whether to overwrite if destination exists
        
    Returns:
        True if copied successfully, False otherwise
        
    Examples:
        >>> copy_file('source.txt', 'dest.txt')
        True
    """
    try:
        if not os.path.exists(source):
            logger.error(f"Source file does not exist: {source}")
            return False
        
        if os.path.exists(destination) and not overwrite:
            logger.warning(f"Destination file exists: {destination}")
            return False
        
        import shutil
        shutil.copy2(source, destination)
        logger.info(f"Copied {source} to {destination}")
        return True
    except Exception as e:
        logger.error(f"Failed to copy file: {e}")
        return False


def move_file(source: str, destination: str, overwrite: bool = False) -> bool:
    """
    Move a file from source to destination.
    
    Args:
        source: Source file path
        destination: Destination file path
        overwrite: Whether to overwrite if destination exists
        
    Returns:
        True if moved successfully, False otherwise
        
    Examples:
        >>> move_file('source.txt', 'dest.txt')
        True
    """
    try:
        if not os.path.exists(source):
            logger.error(f"Source file does not exist: {source}")
            return False
        
        if os.path.exists(destination) and not overwrite:
            logger.warning(f"Destination file exists: {destination}")
            return False
        
        import shutil
        shutil.move(source, destination)
        logger.info(f"Moved {source} to {destination}")
        return True
    except Exception as e:
        logger.error(f"Failed to move file: {e}")
        return False
