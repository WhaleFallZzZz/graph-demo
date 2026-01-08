"""
统一的配置管理工具模块

提供统一的配置读取、验证和管理功能
"""

import os
import logging
from typing import Any, Optional, Dict, Type, Union, List
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigManager:
    """统一配置管理器"""
    
    def __init__(self, env_prefix: str = ""):
        """
        初始化配置管理器
        
        Args:
            env_prefix: 环境变量前缀
        """
        self.env_prefix = env_prefix
        self._cache: Dict[str, Any] = {}
    
    def _get_env_key(self, key: str) -> str:
        """获取带前缀的环境变量键名"""
        return f"{self.env_prefix}{key}" if self.env_prefix else key
    
    def get(
        self,
        key: str,
        default: Any = None,
        required: bool = False,
        value_type: Optional[Type] = None,
        choices: Optional[List[Any]] = None
    ) -> Any:
        """
        获取配置值
        
        Args:
            key: 配置键名
            default: 默认值
            required: 是否必需
            value_type: 值类型（int, float, bool, str等）
            choices: 可选值列表
            
        Returns:
            配置值
            
        Raises:
            ValueError: 当必需的配置项缺失时
        """
        if key in self._cache:
            return self._cache[key]
        
        env_key = self._get_env_key(key)
        value = os.getenv(env_key, default)
        
        if value is None and required:
            raise ValueError(f"Required configuration '{key}' (env: {env_key}) is not set")
        
        if value is not None:
            if value_type is not None:
                try:
                    if value_type == bool:
                        value = str(value).lower() in ('true', '1', 'yes', 'on')
                    else:
                        value = value_type(value)
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Configuration '{key}' must be of type {value_type.__name__}: {e}")
            
            if choices is not None and value not in choices:
                raise ValueError(f"Configuration '{key}' must be one of {choices}, got: {value}")
        
        self._cache[key] = value
        return value
    
    def get_int(self, key: str, default: int = 0, required: bool = False) -> int:
        """获取整数配置"""
        return self.get(key, default, required, int)
    
    def get_float(self, key: str, default: float = 0.0, required: bool = False) -> float:
        """获取浮点数配置"""
        return self.get(key, default, required, float)
    
    def get_bool(self, key: str, default: bool = False, required: bool = False) -> bool:
        """获取布尔值配置"""
        return self.get(key, default, required, bool)
    
    def get_str(self, key: str, default: str = "", required: bool = False) -> str:
        """获取字符串配置"""
        return self.get(key, default, required, str)
    
    def get_list(self, key: str, default: Optional[List[Any]] = None, separator: str = ",") -> List[str]:
        """
        获取列表配置
        
        Args:
            key: 配置键名
            default: 默认值
            separator: 分隔符
            
        Returns:
            列表值
        """
        if default is None:
            default = []
        
        value = self.get_str(key)
        if not value:
            return default
        
        return [item.strip() for item in value.split(separator) if item.strip()]
    
    def get_path(self, key: str, default: str = "", required: bool = False, must_exist: bool = False) -> Path:
        """
        获取路径配置
        
        Args:
            key: 配置键名
            default: 默认值
            required: 是否必需
            must_exist: 路径是否必须存在
            
        Returns:
            Path对象
        """
        path_str = self.get_str(key, default, required)
        path = Path(path_str).expanduser().absolute()
        
        if must_exist and not path.exists():
            raise ValueError(f"Path '{path}' does not exist")
        
        return path
    
    def set(self, key: str, value: Any) -> None:
        """
        设置配置值（仅内存中）
        
        Args:
            key: 配置键名
            value: 配置值
        """
        self._cache[key] = value
    
    def clear_cache(self) -> None:
        """清除配置缓存"""
        self._cache.clear()
    
    def load_from_dict(self, config_dict: Dict[str, Any], prefix: str = "") -> None:
        """
        从字典加载配置
        
        Args:
            config_dict: 配置字典
            prefix: 配置键前缀
        """
        for key, value in config_dict.items():
            full_key = f"{prefix}{key}" if prefix else key
            self._cache[full_key] = value
    
    def load_from_env_file(self, env_file: str = ".env") -> None:
        """
        从.env文件加载配置
        
        Args:
            env_file: .env文件路径
        """
        env_path = Path(env_file)
        if not env_path.exists():
            logger.warning(f"Environment file '{env_file}' not found")
            return
        
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                
                if key not in os.environ:
                    os.environ[key] = value
                    logger.debug(f"Loaded environment variable: {key}")


class NestedConfigManager:
    """嵌套配置管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化嵌套配置管理器
        
        Args:
            config: 配置字典
        """
        self._config = config
    
    def get(self, key: str, default: Any = None, required: bool = False) -> Any:
        """
        获取嵌套配置值（支持点号分隔的路径）
        
        Args:
            key: 配置键路径（如 "api.timeout"）
            default: 默认值
            required: 是否必需
            
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                if required:
                    raise ValueError(f"Required configuration '{key}' not found")
                return default
        
        return value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        获取配置节
        
        Args:
            section: 节名
            
        Returns:
            配置字典
        """
        return self.get(section, {})
    
    def update(self, key: str, value: Any) -> None:
        """
        更新配置值
        
        Args:
            key: 配置键路径
            value: 新值
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self._config.copy()


def validate_config(config: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """
    验证配置是否符合模式
    
    Args:
        config: 配置字典
        schema: 配置模式
        
    Returns:
        是否验证通过
        
    Raises:
        ValueError: 当配置不符合模式时
    """
    for key, spec in schema.items():
        if key not in config:
            if spec.get('required', False):
                raise ValueError(f"Required configuration '{key}' is missing")
            continue
        
        value = config[key]
        expected_type = spec.get('type')
        
        if expected_type and not isinstance(value, expected_type):
            raise ValueError(f"Configuration '{key}' must be of type {expected_type.__name__}, got {type(value).__name__}")
        
        choices = spec.get('choices')
        if choices and value not in choices:
            raise ValueError(f"Configuration '{key}' must be one of {choices}, got {value}")
    
    return True


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并多个配置字典（后面的覆盖前面的）
    
    Args:
        *configs: 配置字典
        
    Returns:
        合并后的配置字典
    """
    result = {}
    for config in configs:
        result.update(config)
    return result
