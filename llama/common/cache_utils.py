"""
知识图谱系统的缓存管理工具。

本模块提供统一的缓存功能，包括：
- 带大小限制的 LRU 缓存
- 持久化缓存到磁盘
- 缓存统计和指标
- 缓存失效策略
- 线程安全操作
"""

import os
import json
import pickle
import hashlib
import logging
import threading
from typing import Any, Optional, Dict, List, Tuple, Callable
from datetime import datetime, timedelta
from collections import OrderedDict
from pathlib import Path

logger = logging.getLogger(__name__)


class LRUCache:
    """
    线程安全的 LRU（最近最少使用）缓存实现。
    
    当达到容量限制时，自动淘汰最近最少使用的项目。
    """
    
    def __init__(self, capacity: int = 1000):
        """
        初始化 LRU 缓存。
        
        Args:
            capacity: 最大存储项目数
        """
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        
        self.capacity = capacity
        self.cache: OrderedDict = OrderedDict()
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def get(self, key: Any) -> Optional[Any]:
        """
        从缓存中获取项目。
        
        Args:
            key: 缓存键
            
        Returns:
            缓存的值，如果未找到则返回 None
        """
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None
            
            # Move to end (most recently used)
            value = self.cache.pop(key)
            self.cache[key] = value
            self.hits += 1
            
            return value
    
    def put(self, key: Any, value: Any) -> None:
        """
        将项目放入缓存。
        
        Args:
            key: 缓存键
            value: 要缓存的值
        """
        with self.lock:
            # Update existing key
            if key in self.cache:
                self.cache.pop(key)
                self.cache[key] = value
                return
            
            # Check capacity and evict if necessary
            if len(self.cache) >= self.capacity:
                oldest_key = next(iter(self.cache))
                self.cache.pop(oldest_key)
                self.evictions += 1
                logger.debug(f"Evicted key: {oldest_key}")
            
            # Add new key
            self.cache[key] = value
    
    def remove(self, key: Any) -> bool:
        """
        从缓存中移除项目。
        
        Args:
            key: 缓存键
            
        Returns:
            如果移除成功返回 True，如果未找到返回 False
        """
        with self.lock:
            if key in self.cache:
                self.cache.pop(key)
                return True
            return False
    
    def clear(self) -> None:
        """清除缓存中的所有项目。"""
        with self.lock:
            self.cache.clear()
            logger.info("Cache cleared")
    
    def size(self) -> int:
        """获取当前缓存大小。"""
        with self.lock:
            return len(self.cache)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息。
        
        Returns:
            包含缓存指标的字典
        """
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0
            
            return {
                'size': len(self.cache),
                'capacity': self.capacity,
                'hits': self.hits,
                'misses': self.misses,
                'evictions': self.evictions,
                'hit_rate': hit_rate,
                'usage': len(self.cache) / self.capacity
            }
    
    def reset_stats(self) -> None:
        """重置缓存统计信息。"""
        with self.lock:
            self.hits = 0
            self.misses = 0
            self.evictions = 0


class TTLCache:
    """
    生存时间（TTL）缓存实现。
    
    项目在 TTL 过期后自动被淘汰。
    """
    
    def __init__(self, ttl_seconds: int = 3600):
        """
        初始化 TTL 缓存。
        
        Args:
            ttl_seconds: 默认生存时间（秒，默认：1小时）
        """
        if ttl_seconds <= 0:
            raise ValueError("TTL must be positive")
        
        self.default_ttl = ttl_seconds
        self.cache: Dict[Any, Tuple[Any, datetime]] = {}
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.expirations = 0
    
    def get(self, key: Any) -> Optional[Any]:
        """
        从缓存中获取项目（如果未过期）。
        
        Args:
            key: 缓存键
            
        Returns:
            缓存的值，如果未找到或已过期则返回 None
        """
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None
            
            value, expiry_time = self.cache[key]
            
            # Check if expired
            if datetime.now() > expiry_time:
                del self.cache[key]
                self.expirations += 1
                logger.debug(f"Key expired: {key}")
                return None
            
            self.hits += 1
            return value
    
    def put(self, key: Any, value: Any, ttl: Optional[int] = None) -> None:
        """
        将项目放入缓存并设置 TTL。
        
        Args:
            key: 缓存键
            value: 要缓存的值
            ttl: 生存时间（秒，如果为 None 则使用默认值）
        """
        if ttl is None:
            ttl = self.default_ttl
        
        expiry_time = datetime.now() + timedelta(seconds=ttl)
        
        with self.lock:
            self.cache[key] = (value, expiry_time)
    
    def remove(self, key: Any) -> bool:
        """
        从缓存中移除项目。
        
        Args:
            key: 缓存键
            
        Returns:
            如果移除成功返回 True，如果未找到返回 False
        """
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """清除缓存中的所有项目。"""
        with self.lock:
            self.cache.clear()
            logger.info("TTL cache cleared")
    
    def cleanup_expired(self) -> int:
        """
        从缓存中移除所有已过期的项目。
        
        Returns:
            移除的项目数量
        """
        with self.lock:
            now = datetime.now()
            expired_keys = [
                key for key, (_, expiry) in self.cache.items()
                if now > expiry
            ]
            
            for key in expired_keys:
                del self.cache[key]
                self.expirations += 1
            
            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired items")
            
            return len(expired_keys)
    
    def size(self) -> int:
        """获取当前缓存大小。"""
        with self.lock:
            return len(self.cache)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息。
        
        Returns:
            包含缓存指标的字典
        """
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0
            
            return {
                'size': len(self.cache),
                'default_ttl': self.default_ttl,
                'hits': self.hits,
                'misses': self.misses,
                'expirations': self.expirations,
                'hit_rate': hit_rate
            }


class PersistentCache:
    """
    持久化缓存，将数据存储在磁盘上。
    
    支持 JSON 和 pickle 序列化。
    """
    
    def __init__(self, 
                 cache_dir: str = '.cache',
                 use_pickle: bool = False):
        """
        初始化持久化缓存。
        
        Args:
            cache_dir: 存储缓存文件的目录
            use_pickle: 使用 pickle 序列化（默认：JSON）
        """
        self.cache_dir = Path(cache_dir)
        self.use_pickle = use_pickle
        self.lock = threading.RLock()
        
        # 创建缓存目录
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"持久化缓存初始化于: {self.cache_dir}")
    
    def _get_cache_path(self, key: str) -> Path:
        """
        获取缓存键对应的文件路径。
        
        Args:
            key: 缓存键
            
        Returns:
            缓存文件的路径
        """
        # 对键进行哈希以获得安全的文件名
        key_hash = hashlib.md5(key.encode()).hexdigest()
        ext = '.pkl' if self.use_pickle else '.json'
        return self.cache_dir / f"{key_hash}{ext}"
    
    def get(self, key: str) -> Optional[Any]:
        """
        从持久化缓存中获取项目。
        
        Args:
            key: 缓存键
            
        Returns:
            缓存的值，如果未找到则返回 None
        """
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return None
        
        try:
            with self.lock:
                if self.use_pickle:
                    with open(cache_path, 'rb') as f:
                        return pickle.load(f)
                else:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read cache file {cache_path}: {e}")
            return None
    
    def put(self, key: str, value: Any) -> bool:
        """
        将项目放入持久化缓存。
        
        Args:
            key: 缓存键
            value: 要缓存的值
            
        Returns:
            如果成功返回 True，否则返回 False
        """
        cache_path = self._get_cache_path(key)
        
        try:
            with self.lock:
                if self.use_pickle:
                    with open(cache_path, 'wb') as f:
                        pickle.dump(value, f)
                else:
                    with open(cache_path, 'w', encoding='utf-8') as f:
                        json.dump(value, f, ensure_ascii=False, indent=2)
                
                logger.debug(f"Cached value for key: {key}")
                return True
        except Exception as e:
            logger.error(f"Failed to write cache file {cache_path}: {e}")
            return False
    
    def remove(self, key: str) -> bool:
        """
        从持久化缓存中移除项目。
        
        Args:
            key: 缓存键
            
        Returns:
            如果移除成功返回 True，如果未找到返回 False
        """
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return False
        
        try:
            with self.lock:
                cache_path.unlink()
                logger.debug(f"Removed cache for key: {key}")
                return True
        except Exception as e:
            logger.error(f"Failed to remove cache file {cache_path}: {e}")
            return False
    
    def clear(self) -> None:
        """清除持久化缓存中的所有项目。"""
        try:
            with self.lock:
                for cache_file in self.cache_dir.glob('*'):
                    cache_file.unlink()
                
                logger.info(f"已清除持久化缓存: {self.cache_dir}")
        except Exception as e:
            logger.error(f"清除缓存失败: {e}")
    
    def size(self) -> int:
        """获取缓存项目数量。"""
        try:
            return len(list(self.cache_dir.glob('*')))
        except Exception:
            return 0
    
    def get_size_bytes(self) -> int:
        """获取缓存总大小（字节）。"""
        try:
            total_size = sum(f.stat().st_size for f in self.cache_dir.glob('*'))
            return total_size
        except Exception:
            return 0


class CacheManager:
    """
    统一缓存管理器，结合多种缓存策略。
    
    提供简单的缓存接口，支持自动回退和统计跟踪。
    """
    
    def __init__(self,
                 lru_capacity: int = 1000,
                 ttl_seconds: int = 3600,
                 enable_persistent: bool = True,
                 cache_dir: str = '.cache'):
        """
        初始化缓存管理器。
        
        Args:
            lru_capacity: LRU 缓存容量
            ttl_seconds: TTL 缓存的默认 TTL
            enable_persistent: 启用持久化缓存到磁盘
            cache_dir: 持久化缓存的目录
        """
        self.lru_cache = LRUCache(capacity=lru_capacity)
        self.ttl_cache = TTLCache(ttl_seconds=ttl_seconds)
        self.persistent_cache = PersistentCache(
            cache_dir=cache_dir,
            use_pickle=False
        ) if enable_persistent else None
        
        self.enable_persistent = enable_persistent
        logger.info("Cache manager initialized")
    
    def get(self, key: str, 
              use_lru: bool = True,
              use_ttl: bool = True,
              use_persistent: bool = True) -> Optional[Any]:
        """
        从缓存中获取项目，使用回退策略。
        
        Args:
            key: 缓存键
            use_lru: 检查 LRU 缓存
            use_ttl: 检查 TTL 缓存
            use_persistent: 检查持久化缓存
            
        Returns:
            缓存的值，如果未找到则返回 None
        """
        # 首先尝试 LRU 缓存（最快）
        if use_lru:
            value = self.lru_cache.get(key)
            if value is not None:
                return value
        
        # 尝试 TTL 缓存
        if use_ttl:
            value = self.ttl_cache.get(key)
            if value is not None:
                # 填充 LRU 缓存以便下次更快访问
                self.lru_cache.put(key, value)
                return value
        
        # 尝试持久化缓存（最慢）
        if use_persistent and self.persistent_cache:
            value = self.persistent_cache.get(key)
            if value is not None:
                # 填充其他缓存以便更快访问
                self.lru_cache.put(key, value)
                self.ttl_cache.put(key, value)
                return value
        
        return None
    
    def put(self, key: str, 
              value: Any,
              ttl: Optional[int] = None) -> None:
        """
        将项目放入所有启用的缓存中。
        
        Args:
            key: 缓存键
            value: 要缓存的值
            ttl: TTL 缓存的 TTL（如果为 None 则使用默认值）
        """
        self.lru_cache.put(key, value)
        self.ttl_cache.put(key, value, ttl=ttl)
        
        if self.enable_persistent and self.persistent_cache:
            self.persistent_cache.put(key, value)
    
    def remove(self, key: str) -> None:
        """
        从所有缓存中移除项目。
        
        Args:
            key: 缓存键
        """
        self.lru_cache.remove(key)
        self.ttl_cache.remove(key)
        
        if self.enable_persistent and self.persistent_cache:
            self.persistent_cache.remove(key)
    
    def clear(self, 
              clear_lru: bool = True,
              clear_ttl: bool = True,
              clear_persistent: bool = True) -> None:
        """
        清除缓存。
        
        Args:
            clear_lru: 清除 LRU 缓存
            clear_ttl: 清除 TTL 缓存
            clear_persistent: 清除持久化缓存
        """
        if clear_lru:
            self.lru_cache.clear()
        
        if clear_ttl:
            self.ttl_cache.clear()
        
        if clear_persistent and self.enable_persistent and self.persistent_cache:
            self.persistent_cache.clear()
    
    def cleanup(self) -> None:
        """执行清理操作（过期项目等）。"""
        self.ttl_cache.cleanup_expired()
        logger.info("缓存清理完成")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取综合缓存统计信息。
        
        Returns:
            包含所有缓存指标的字典
        """
        stats = {
            'lru_cache': self.lru_cache.get_stats(),
            'ttl_cache': self.ttl_cache.get_stats()
        }
        
        if self.enable_persistent and self.persistent_cache:
            stats['persistent_cache'] = {
                'size': self.persistent_cache.size(),
                'size_bytes': self.persistent_cache.get_size_bytes()
            }
        
        return stats


def cached(ttl: Optional[int] = None,
           cache_manager: Optional[CacheManager] = None) -> Callable:
    """
    用于缓存函数结果的装饰器。
    
    Args:
        ttl: 生存时间（秒），None 表示无 TTL
        cache_manager: 缓存管理器实例（如果为 None 则创建默认实例）
        
    Returns:
        带有缓存的装饰函数
        
    Examples:
        >>> @cached(ttl=3600)
        ... def expensive_function(x):
        ...     return x * 2
    """
    if cache_manager is None:
        cache_manager = CacheManager()
    
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # 从函数名称和参数创建缓存键
            key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # 尝试从缓存获取
            cached_value = cache_manager.get(key)
            if cached_value is not None:
                return cached_value
            
            # 执行函数并缓存结果
            result = func(*args, **kwargs)
            cache_manager.put(key, result, ttl=ttl)
            
            return result
        
        return wrapper
    
    return decorator
