#!/usr/bin/env python3
"""
简化版的文件类型检测模块 - 仅使用扩展名和MIME类型检测
"""

from ast import dump
import os
import mimetypes
from pathlib import Path
from typing import Optional, Dict, Any, Set
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class SimpleFileTypeDetector:
    """简化的文件类型检测器 - 仅使用扩展名和MIME类型"""
    
    def __init__(self):
        # 支持的文件扩展名
        self.allowed_extensions = {
            # 文本文件
            'txt', 'md', 'rst', 'text',
            # Word文档
            'doc', 'docx',
            # PDF
            'pdf',
            # 网页文件
            'html', 'htm', 'xml',
            # 代码文件
            'py', 'js', 'java', 'cpp', 'c', 'h', 'css', 'json', 'yaml', 'yml'
        }
        
        # MIME类型映射
        self.mime_type_mapping = {
            'text/plain': 'txt',
            'text/html': 'html',
            'text/xml': 'xml',
            'application/msword': 'doc',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
            'application/pdf': 'pdf',
            'application/json': 'json',
            'text/x-python': 'py',
            'text/javascript': 'js',
            'text/css': 'css',
            'text/markdown': 'md',
            'text/x-yaml': 'yaml'
        }
    
    def detect_by_extension(self, filename: str) -> Optional[str]:
        """通过文件扩展名检测文件类型"""
        if not filename:
            return None
            
        ext = Path(filename).suffix.lower().lstrip('.')
        # 兼容处理：如果文件名包含特殊字符导致扩展名识别错误
        # 例如 "文件名.pdf" 可能被识别为 "pdf"
        # 如果 "文件名.tar.gz" 可能被识别为 "gz"
        
        # 强制检查常见的扩展名
        if filename.lower().endswith('.pdf'):
            return 'pdf'
        elif filename.lower().endswith('.docx'):
            return 'docx'
        elif filename.lower().endswith('.doc'):
            return 'doc'
        elif filename.lower().endswith('.txt'):
            return 'txt'
            
        if ext in self.allowed_extensions:
            return ext
            
        # 检查别名
        extension_aliases = {
            'jpeg': 'jpg',
            'htm': 'html',
            'yml': 'yaml'
        }
        
        return extension_aliases.get(ext)
    
    def detect_by_mime_type(self, file_path: str) -> Optional[str]:
        """通过MIME类型检测文件类型"""
        try:
            mime_type, _ = mimetypes.guess_type(file_path)
            
            if mime_type:
                # 检查映射表
                if mime_type in self.mime_type_mapping:
                    return self.mime_type_mapping[mime_type]
                    
                # 通用的文本文件处理
                if mime_type.startswith('text/'):
                    return 'txt'
                    
        except Exception as e:
            logger.error(f"MIME类型检测失败: {e}")
            
        return None
    
    def detect_file_type(self, file_path: str, filename: Optional[str] = None) -> Dict[str, Any]:
        """综合检测文件类型"""
        if not os.path.exists(file_path):
            return {'error': '文件不存在', 'type': None, 'method': None}
            
        # 使用提供的文件名或从路径获取
        actual_filename = filename or os.path.basename(file_path)
        
        detection_methods = []
        detected_type = None
        confidence = 0
        
        # 方法1：扩展名检测（最快）
        ext_type = self.detect_by_extension(actual_filename)
        if ext_type:
            detection_methods.append('extension')
            detected_type = ext_type
            confidence += 3  # 扩展名检测置信度较高
        
        # 方法2：MIME类型检测
        if not detected_type:
            mime_type = self.detect_by_mime_type(file_path)
            if mime_type:
                detection_methods.append('mime')
                detected_type = mime_type
                confidence += 2
        # 验证检测到的类型是否在允许列表中
        if detected_type and detected_type not in self.allowed_extensions:
            logger.warning(f"检测到的文件类型 '{detected_type}' 不在允许列表中")
            return {
                'type': None, 
                'detected': detected_type,
                'method': detection_methods,
                'confidence': confidence,
                'error': f'不支持的文件类型: {detected_type}'
            }
        
        return {
            'type': detected_type,
            'filename': actual_filename,
            'method': detection_methods,
            'confidence': confidence,
            'allowed': detected_type in self.allowed_extensions if detected_type else False
        }
    
    def is_allowed_file(self, file_path: str, filename: Optional[str] = None) -> bool:
        """判断文件是否允许处理"""
        result = self.detect_file_type(file_path, filename)
        return result.get('allowed', False)
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """获取详细的文件信息"""
        if not os.path.exists(file_path):
            return {'error': '文件不存在'}
            
        file_stat = os.stat(file_path)
        
        # 检测文件类型
        type_result = self.detect_file_type(file_path)
        
        return {
            'path': file_path,
            'size': file_stat.st_size,
            'modified': datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
            'created': datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
            'type_detection': type_result,
            'readable': os.access(file_path, os.R_OK)
        }

# 全局文件类型检测器实例
file_detector = SimpleFileTypeDetector()

# 便捷的包装函数
def detect_file_type(file_path: str, filename: Optional[str] = None) -> Dict[str, Any]:
    """检测文件类型"""
    return file_detector.detect_file_type(file_path, filename)

def is_allowed_file(file_path: str, filename: Optional[str] = None) -> bool:
    """判断文件是否允许处理"""
    return file_detector.is_allowed_file(file_path, filename)

def get_file_info(file_path: str) -> Dict[str, Any]:
    """获取文件详细信息"""
    return file_detector.get_file_info(file_path)