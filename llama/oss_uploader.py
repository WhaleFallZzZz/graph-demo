"""
腾讯云 COS 文件上传模块

提供腾讯云 COS（Cloud Object Storage）文件上传功能，支持文件元数据提取、
MD5 校验、文件类型识别等功能。

主要功能：
- 文件上传到腾讯云 COS
- 文件元数据提取（文件名、大小、扩展名、MIME 类型等）
- MD5 校验确保文件完整性
- 文件类型识别（图片、文档、音频、视频等）
- 自动生成文件访问 URL

使用示例：
    from llama.oss_uploader import COSUploader, OSSConfig
    
    # 创建配置
    config_dict = {
        'cos_secret_id': 'your_secret_id',
        'cos_secret_key': 'your_secret_key',
        'cos_bucket': 'your-bucket',
        'cos_region': 'ap-guangzhou',
        'cos_path': 'uploads'
    }
    
    # 创建上传器
    config = OSSConfig(config_dict)
    uploader = COSUploader(config)
    
    # 上传文件
    with open('file.pdf', 'rb') as f:
        file_data = f.read()
    result = uploader.upload_file(file_data, 'file.pdf')
    print(result['file_url'])
"""

import os
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any, BinaryIO
from pathlib import Path
import logging
import mimetypes

logger = logging.getLogger(__name__)

class OSSConfig:
    """
    OSS 配置类 - 腾讯云 COS 配置
    
    用于存储腾讯云 COS 的连接配置信息。
    
    Attributes:
        drive (str): 存储驱动类型，固定为 'cos'
        cos_secret_id (str): 腾讯云 COS Secret ID
        cos_secret_key (str): 腾讯云 COS Secret Key
        cos_bucket (str): COS 存储桶名称
        cos_region (str): COS 区域，如 'ap-guangzhou'
        cos_path (str): COS 存储路径前缀，默认为 'uploads'
    """
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        初始化 OSS 配置
        
        Args:
            config_dict: 配置字典，包含以下键：
                - cos_secret_id: 腾讯云 COS Secret ID
                - cos_secret_key: 腾讯云 COS Secret Key
                - cos_bucket: COS 存储桶名称
                - cos_region: COS 区域（可选）
                - cos_path: COS 存储路径前缀（可选，默认为 'uploads'）
        """
        self.drive = 'cos'
        
        self.cos_secret_id = config_dict.get('cos_secret_id', '')
        self.cos_secret_key = config_dict.get('cos_secret_key', '')
        self.cos_bucket = config_dict.get('cos_bucket', '')
        self.cos_region = config_dict.get('cos_region', '')
        self.cos_path = config_dict.get('cos_path', 'uploads')


class FileMeta:
    """
    文件元数据类
    
    用于存储和提取文件的元数据信息，包括文件名、大小、扩展名、
    MD5 哈希值、文件类型和 MIME 类型等。
    
    Attributes:
        filename (str): 文件名
        size (int): 文件大小（字节）
        ext (str): 文件扩展名（不含点号）
        md5 (str): 文件的 MD5 哈希值（32 位十六进制字符串）
        kind (str): 文件类型分类（image/document/audio/video/other）
        mime_type (str): 文件的 MIME 类型
    """
    
    def __init__(self, filename: str, size: int, file_data: bytes):
        """
        初始化文件元数据
        
        Args:
            filename: 文件名
            size: 文件大小（字节）
            file_data: 文件数据（用于计算 MD5）
        """
        self.filename = filename
        self.size = size
        self.ext = self._get_extension(filename)
        self.md5 = self._calculate_md5(file_data)
        self.kind = self._get_file_kind(self.ext)
        self.mime_type = self._get_mime_type(self.ext)
        
    def _get_extension(self, filename: str) -> str:
        """
        获取文件扩展名
        
        Args:
            filename: 文件名
            
        Returns:
            文件扩展名（不含点号，小写），如果没有扩展名则返回空字符串
        """
        return Path(filename).suffix.lower()[1:] if '.' in filename else ''
        
    def _calculate_md5(self, data: bytes) -> str:
        """
        计算 MD5 哈希值
        
        Args:
            data: 文件数据
            
        Returns:
            32 位十六进制 MD5 哈希字符串
        """
        return hashlib.md5(data).hexdigest()
        
    def _get_file_kind(self, ext: str) -> str:
        """
        获取文件类型分类
        
        根据文件扩展名判断文件类型，返回预定义的分类之一。
        
        Args:
            ext: 文件扩展名
            
        Returns:
            文件类型分类：
            - 'image': 图片文件
            - 'document': 文档文件
            - 'audio': 音频文件
            - 'video': 视频文件
            - 'other': 其他类型
        """
        image_exts = {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'svg'}
        doc_exts = {'doc', 'docx', 'pdf', 'txt', 'xls', 'xlsx', 'ppt', 'pptx'}
        audio_exts = {'mp3', 'wav', 'flac', 'aac', 'm4a'}
        video_exts = {'mp4', 'avi', 'mov', 'wmv', 'flv', 'mkv'}
        
        if ext in image_exts:
            return 'image'
        elif ext in doc_exts:
            return 'document'
        elif ext in audio_exts:
            return 'audio'
        elif ext in video_exts:
            return 'video'
        else:
            return 'other'
            
    def _get_mime_type(self, ext: str) -> str:
        """
        获取 MIME 类型
        
        根据文件扩展名猜测文件的 MIME 类型。
        
        Args:
            ext: 文件扩展名
            
        Returns:
            MIME 类型字符串，如果无法识别则返回 'application/octet-stream'
        """
        mime_type, _ = mimetypes.guess_type(f'test.{ext}')
        return mime_type or 'application/octet-stream'


class COSUploader:
    """
    腾讯云 COS 上传器
    
    提供文件上传到腾讯云 COS 的功能，支持文件元数据提取、
    MD5 校验和自动生成访问 URL。
    
    上传流程：
    1. 验证文件数据和文件名
    2. 创建文件元数据（计算 MD5、识别文件类型等）
    3. 检查文件大小限制
    4. 上传到腾讯云 COS
    5. 返回上传结果和访问 URL
    
    文件命名规则：
    - 如果有扩展名：{cos_path}/{date}/{md5前16位}.{ext}
    - 如果无扩展名：{cos_path}/{date}/{md5前16位}_{filename}
    
    Attributes:
        config (OSSConfig): COS 配置对象
    """
    
    def __init__(self, config: OSSConfig):
        """
        初始化 COS 上传器
        
        Args:
            config: OSSConfig 配置对象，包含 COS 的连接信息
        """
        self.config = config
        
    def upload_file(self, file_data: bytes, filename: str, 
                   max_size: int = 200 * 1024 * 1024) -> Dict[str, Any]:
        """
        上传文件到腾讯云 COS
        
        将文件数据上传到腾讯云 COS，并返回上传结果和访问 URL。
        文件会自动命名以避免中文文件名导致的 URL 问题。
        
        Args:
            file_data: 文件二进制数据
            filename: 原始文件名
            max_size: 最大文件大小限制（字节），默认为 200MB
            
        Returns:
            Dict[str, Any]: 上传结果字典，包含：
                - success (bool): 上传是否成功
                - drive (str): 存储驱动类型，固定为 'cos'
                - filename (str): 原始文件名
                - file_url (str): 文件访问 URL
                - object_key (str): COS 对象键
                - etag (str): COS 返回的 ETag
                - size (int): 文件大小（字节）
                - md5 (str): 文件 MD5 哈希值
                - ext (str): 文件扩展名
                - mime_type (str): 文件 MIME 类型
            
        Raises:
            ValueError: 文件数据为空、文件名为空或文件大小超过限制
            Exception: 上传失败
        """
        if not file_data:
            raise ValueError("文件数据为空")
        
        if not filename:
            raise ValueError("文件名不能为空")
        
        try:
            # 创建文件元数据
            meta = FileMeta(filename, len(file_data), file_data)
            logger.info(f"准备上传文件: {filename}, 大小: {meta.size} bytes, MD5: {meta.md5[:8]}...")
            
            # 检查文件大小
            if meta.size > max_size:
                raise ValueError(f"文件大小 {meta.size} 超过限制 {max_size} bytes")
            
            # 上传到腾讯云COS
            return self._upload_cos(file_data, meta)
                
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"文件上传失败: {str(e)}", exc_info=True)
            raise
    
    def _upload_cos(self, file_data: bytes, meta: FileMeta) -> Dict[str, Any]:
        """
        腾讯云 COS 上传实现
        
        使用腾讯云 COS SDK 将文件上传到指定的存储桶。
        
        上传步骤：
        1. 创建 COS 客户端
        2. 生成对象键（文件路径）
        3. 上传文件数据
        4. 生成访问 URL
        5. 返回上传结果
        
        Args:
            file_data: 文件二进制数据
            meta: 文件元数据对象
            
        Returns:
            Dict[str, Any]: 上传结果字典
            
        Raises:
            ImportError: 未安装 cos-python-sdk-v5 库
            Exception: COS 上传失败
        """
        try:
            # 这里需要安装 cos-python-sdk-v5 库: pip install cos-python-sdk-v5
            from qcloud_cos import CosConfig, CosS3Client
            
            # 创建 COS 客户端
            config = CosConfig(
                Region=self.config.cos_region,
                SecretId=self.config.cos_secret_id,
                SecretKey=self.config.cos_secret_key
            )
            client = CosS3Client(config)
            
            # 生成对象键
            now_date = datetime.now().strftime("%Y-%m-%d")
            # 使用 hash.ext 格式，避免中文文件名导致的 URL 问题，也符合用户预期的格式
            if meta.ext:
                object_key = f"{self.config.cos_path}/{now_date}/{meta.md5[:16]}.{meta.ext}"
            else:
                # 如果没有扩展名，则使用 md5_filename
                object_key = f"{self.config.cos_path}/{now_date}/{meta.md5[:16]}_{meta.filename}"
            
            # 上传文件
            response = client.put_object(
                Bucket=self.config.cos_bucket,
                Body=file_data,
                Key=object_key,
                ContentType=meta.mime_type
            )
            
            # 生成访问 URL
            file_url = f"https://{self.config.cos_bucket}.cos.{self.config.cos_region}.myqcloud.com/{object_key}"
            
            return {
                'success': True,
                'drive': 'cos',
                'filename': meta.filename,
                'file_url': file_url,
                'object_key': object_key,
                'etag': response.get('ETag', ''),
                'size': meta.size,
                'md5': meta.md5,
                'ext': meta.ext,
                'mime_type': meta.mime_type
            }
            
        except ImportError:
            raise ImportError("请安装 cos-python-sdk-v5 库: pip install cos-python-sdk-v5")
        except Exception as e:
            logger.error(f"COS 上传失败: {e}")
            raise


# 全局 COS 上传器实例
_uploader_instance = None


def get_uploader(config: Optional[Dict[str, Any]] = None) -> COSUploader:
    """
    获取腾讯云 COS 上传器实例（单例模式）
    
    使用单例模式确保全局只有一个 COS 上传器实例。
    第一次调用时需要传入配置字典，后续调用可以不传参数直接返回已创建的实例。
    
    Args:
        config: 配置字典（仅在第一次调用时需要），包含：
            - cos_secret_id: 腾讯云 COS Secret ID
            - cos_secret_key: 腾讯云 COS Secret Key
            - cos_bucket: COS 存储桶名称
            - cos_region: COS 区域（可选）
            - cos_path: COS 存储路径前缀（可选）
    
    Returns:
        COSUploader: COS 上传器实例
    
    Raises:
        ValueError: 第一次调用时未提供配置
    
    Example:
        # 第一次调用，传入配置
        config = {
            'cos_secret_id': 'your_secret_id',
            'cos_secret_key': 'your_secret_key',
            'cos_bucket': 'your-bucket',
            'cos_region': 'ap-guangzhou'
        }
        uploader = get_uploader(config)
        
        # 后续调用，直接获取实例
        uploader = get_uploader()
    """
    global _uploader_instance
    
    if _uploader_instance is None:
        if config is None:
            raise ValueError("第一次调用 get_uploader() 时必须提供配置参数")
        cos_config = OSSConfig(config)
        _uploader_instance = COSUploader(cos_config)
        logger.info("COS 上传器实例已创建")
    
    return _uploader_instance