"""
腾讯云 COS 文件上传模块
仅支持腾讯云 COS 上传
"""

import os
import hashlib
import datetime
from typing import Optional, Dict, Any, BinaryIO
from pathlib import Path
import logging
import mimetypes

logger = logging.getLogger(__name__)

class OSSConfig:
    """OSS 配置类 - 仅腾讯云COS"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self.drive = 'cos'  # 只使用腾讯云COS
        
        # 腾讯云 COS 配置
        self.cos_secret_id = config_dict.get('cos_secret_id', '')
        self.cos_secret_key = config_dict.get('cos_secret_key', '')
        self.cos_bucket = config_dict.get('cos_bucket', '')
        self.cos_region = config_dict.get('cos_region', '')
        self.cos_path = config_dict.get('cos_path', 'uploads')


class FileMeta:
    """文件元数据"""
    
    def __init__(self, filename: str, size: int, file_data: bytes):
        self.filename = filename
        self.size = size
        self.ext = self._get_extension(filename)
        self.md5 = self._calculate_md5(file_data)
        self.kind = self._get_file_kind(self.ext)
        self.mime_type = self._get_mime_type(self.ext)
        
    def _get_extension(self, filename: str) -> str:
        """获取文件扩展名"""
        return Path(filename).suffix.lower()[1:] if '.' in filename else ''
        
    def _calculate_md5(self, data: bytes) -> str:
        """计算 MD5"""
        return hashlib.md5(data).hexdigest()
        
    def _get_file_kind(self, ext: str) -> str:
        """获取文件类型"""
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
        """获取 MIME 类型"""
        mime_type, _ = mimetypes.guess_type(f'test.{ext}')
        return mime_type or 'application/octet-stream'


class COSUploader:
    """腾讯云 COS 上传器"""
    
    def __init__(self, config: OSSConfig):
        self.config = config
        
    def upload_file(self, file_data: bytes, filename: str, 
                   max_size: int = 100 * 1024 * 1024) -> Dict[str, Any]:
        """
        上传文件到腾讯云COS
        
        Args:
            file_data: 文件数据
            filename: 文件名
            max_size: 最大文件大小 (默认 100MB)
            
        Returns:
            上传结果信息
            
        Raises:
            ValueError: 文件大小超过限制
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
            
            # 只使用腾讯云COS上传
            return self._upload_cos(file_data, meta)
                
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"文件上传失败: {str(e)}", exc_info=True)
            raise
    
    def upload_image(self, file_data: bytes, filename: str,
                    max_size: int = 10 * 1024 * 1024) -> Dict[str, Any]:
        """
        上传图片文件
        
        Args:
            file_data: 图片数据
            filename: 文件名
            max_size: 最大图片大小 (默认 10MB)
            
        Returns:
            上传结果信息
        """
        meta = FileMeta(filename, len(file_data), file_data)
        
        # 验证是否为图片
        if meta.kind != 'image':
            raise ValueError("上传的文件不是图片")
            
        return self.upload_file(file_data, filename, max_size)
    
    def _upload_cos(self, file_data: bytes, meta: FileMeta) -> Dict[str, Any]:
        """腾讯云 COS 上传"""
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
            now_date = datetime.datetime.now().strftime("%Y-%m-%d")
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
    
    # 其他上传方法已移除，只保留腾讯云COS


# 全局COS上传器实例
_uploader_instance = None

def get_uploader(config: Optional[Dict[str, Any]] = None) -> COSUploader:
    """获取腾讯云COS上传器实例"""
    global _uploader_instance
    
    if _uploader_instance is None and config is not None:
        cos_config = OSSConfig(config)
        _uploader_instance = COSUploader(cos_config)
    
    return _uploader_instance