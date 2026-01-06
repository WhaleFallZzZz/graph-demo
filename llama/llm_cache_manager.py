"""
LLM 调用缓存管理器 - 避免重复的LLM调用
使用内容哈希作为key,支持LRU淘汰策略和持久化
"""

import hashlib
import json
import logging
import os
import pickle
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Any, Dict

logger = logging.getLogger(__name__)


class LLMCacheManager:
    """LLM调用缓存管理器"""
    
    def __init__(
        self,
        max_cache_size: int = 1000,
        ttl_seconds: int = 86400,  # 24小时
        enable_persistence: bool = True,
        cache_dir: str = None
    ):
        """
        初始化缓存管理器
        
        Args:
            max_cache_size: 最大缓存条目数
            ttl_seconds: 缓存过期时间(秒)
            enable_persistence: 是否启用持久化
            cache_dir: 缓存目录
        """
        self.max_cache_size = max_cache_size
        self.ttl_seconds = ttl_seconds
        self.enable_persistence = enable_persistence
        
        # 缓存存储 {cache_key: (result, timestamp)}
        self.cache: OrderedDict[str, tuple] = OrderedDict()
        self.lock = threading.RLock()
        
        # 统计信息
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expired": 0
        }
        
        # 持久化配置
        if cache_dir is None:
            cache_dir = os.path.join(os.getcwd(), ".llm_cache")
        self.cache_dir = Path(cache_dir)
        
        if enable_persistence:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_file = self.cache_dir / "llm_cache.pkl"
            self._load_cache()
        
        logger.info(f"LLM缓存管理器初始化: max_size={max_cache_size}, ttl={ttl_seconds}s, persistence={enable_persistence}")
    
    def _generate_cache_key(self, prompt: str, model_params: Dict = None) -> str:
        """
        生成缓存键
        
        Args:
            prompt: LLM输入提示
            model_params: 模型参数(temperature, max_tokens等)
        
        Returns:
            缓存键(MD5哈希)
        """
        # 标准化模型参数
        params_str = ""
        if model_params:
            # 只考虑影响输出的关键参数
            key_params = {
                k: v for k, v in sorted(model_params.items())
                if k in ["temperature", "max_tokens", "top_p", "top_k", "model"]
            }
            params_str = json.dumps(key_params, sort_keys=True)
        
        # 生成哈希
        content = f"{prompt}|||{params_str}"
        cache_key = hashlib.md5(content.encode('utf-8')).hexdigest()
        return cache_key
    
    def get(self, prompt: str, model_params: Dict = None) -> Optional[str]:
        """
        从缓存获取LLM结果
        
        Args:
            prompt: LLM输入提示
            model_params: 模型参数
        
        Returns:
            缓存的LLM输出,如果未命中则返回None
        """
        cache_key = self._generate_cache_key(prompt, model_params)
        
        with self.lock:
            if cache_key in self.cache:
                result, timestamp = self.cache[cache_key]
                
                # 检查是否过期
                if time.time() - timestamp > self.ttl_seconds:
                    # 过期,删除
                    del self.cache[cache_key]
                    self.stats["expired"] += 1
                    self.stats["misses"] += 1
                    logger.debug(f"缓存过期: {cache_key[:8]}...")
                    return None
                
                # 命中,移到末尾(LRU)
                self.cache.move_to_end(cache_key)
                self.stats["hits"] += 1
                
                hit_rate = self.stats["hits"] / (self.stats["hits"] + self.stats["misses"])
                logger.debug(f"缓存命中: {cache_key[:8]}... (命中率: {hit_rate:.2%})")
                return result
            else:
                # 未命中
                self.stats["misses"] += 1
                return None
    
    def put(self, prompt: str, result: str, model_params: Dict = None):
        """
        将LLM结果放入缓存
        
        Args:
            prompt: LLM输入提示
            result: LLM输出结果
            model_params: 模型参数
        """
        cache_key = self._generate_cache_key(prompt, model_params)
        
        with self.lock:
            # 检查是否需要淘汰
            if len(self.cache) >= self.max_cache_size:
                # LRU淘汰:删除最老的条目
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                self.stats["evictions"] += 1
                logger.debug(f"缓存淘汰: {oldest_key[:8]}... (当前大小: {len(self.cache)})")
            
            # 添加到缓存
            self.cache[cache_key] = (result, time.time())
            logger.debug(f"缓存添加: {cache_key[:8]}... (当前大小: {len(self.cache)})")
    
    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            logger.info("缓存已清空")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self.lock:
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0.0
            
            return {
                **self.stats,
                "cache_size": len(self.cache),
                "max_size": self.max_cache_size,
                "hit_rate": hit_rate,
                "total_requests": total_requests
            }
    
    def _load_cache(self):
        """从磁盘加载缓存"""
        if not self.cache_file.exists():
            logger.info("缓存文件不存在,使用空缓存")
            return
        
        try:
            with open(self.cache_file, 'rb') as f:
                loaded_cache = pickle.load(f)
            
            # 过滤过期条目
            current_time = time.time()
            valid_cache = OrderedDict()
            
            for key, (result, timestamp) in loaded_cache.items():
                if current_time - timestamp <= self.ttl_seconds:
                    valid_cache[key] = (result, timestamp)
            
            self.cache = valid_cache
            logger.info(f"从磁盘加载缓存: {len(self.cache)} 条有效记录")
            
        except Exception as e:
            logger.error(f"加载缓存失败: {e}")
    
    def _save_cache(self):
        """保存缓存到磁盘"""
        if not self.enable_persistence:
            return
        
        try:
            with self.lock:
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(self.cache, f)
            logger.info(f"缓存已保存到磁盘: {len(self.cache)} 条记录")
        except Exception as e:
            logger.error(f"保存缓存失败: {e}")
    
    def __del__(self):
        """析构函数,保存缓存"""
        if self.enable_persistence:
            self._save_cache()


# 全局缓存实例
_global_cache: Optional[LLMCacheManager] = None
_cache_lock = threading.Lock()


def get_global_cache() -> LLMCacheManager:
    """获取全局缓存实例(单例模式)"""
    global _global_cache
    
    if _global_cache is None:
        with _cache_lock:
            if _global_cache is None:
                _global_cache = LLMCacheManager(
                    max_cache_size=1000,
                    ttl_seconds=86400,
                    enable_persistence=True
                )
    
    return _global_cache
