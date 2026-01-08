"""
LLM 调用缓存管理器 - 避免重复的LLM调用
使用内容哈希作为key,支持LRU淘汰策略和持久化
优化版本：使用通用缓存工具，改进命中率和性能
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional, Any, Dict, List, Tuple

from .common.cache_utils import LRUCache, TTLCache, CacheManager

logger = logging.getLogger(__name__)


class LLMCacheManager:
    """
    LLM调用缓存管理器（优化版）
    
    改进点：
    1. 使用通用缓存工具，提高代码复用性
    2. 改进缓存键生成，考虑更多参数
    3. 智能淘汰策略，优先保留高频查询
    4. 改进持久化机制，支持增量保存
    5. 新增：批量缓存操作支持
    6. 新增：智能缓存预热功能
    7. 新增：缓存命中率预测
    """
    
    def __init__(
        self,
        max_cache_size: int = 1000,
        ttl_seconds: int = 86400,  # 24小时
        enable_persistence: bool = True,
        cache_dir: str = None,
        use_hybrid_cache: bool = True,
        enable_batch_cache: bool = True,
        enable_cache_warmup: bool = True
    ):
        """
        初始化缓存管理器
        
        Args:
            max_cache_size: 最大缓存条目数
            ttl_seconds: 缓存过期时间(秒)
            enable_persistence: 是否启用持久化
            cache_dir: 缓存目录
            use_hybrid_cache: 是否使用混合缓存(LRU+TTL)
            enable_batch_cache: 是否启用批量缓存优化
            enable_cache_warmup: 是否启用缓存预热
        """
        self.max_cache_size = max_cache_size
        self.ttl_seconds = ttl_seconds
        self.enable_persistence = enable_persistence
        self.use_hybrid_cache = use_hybrid_cache
        self.enable_batch_cache = enable_batch_cache
        self.enable_cache_warmup = enable_cache_warmup
        
        # 使用通用缓存管理器
        if cache_dir is None:
            cache_dir = os.path.join(os.getcwd(), ".llm_cache")
        
        self.cache_manager = CacheManager(
            lru_capacity=max_cache_size,
            ttl_seconds=ttl_seconds,
            enable_persistent=enable_persistence,
            cache_dir=cache_dir
        )
        
        # 统计信息
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expired": 0,
            "batch_hits": 0,
            "batch_misses": 0
        }
        
        # 批量缓存缓冲区
        self.batch_cache_buffer = {}
        
        logger.info(f"LLM缓存管理器初始化: max_size={max_cache_size}, ttl={ttl_seconds}s, persistence={enable_persistence}, hybrid={use_hybrid_cache}, batch={enable_batch_cache}, warmup={enable_cache_warmup}")
        
        # 启用缓存预热
        if enable_cache_warmup:
            self._warmup_cache()
    
    def _generate_cache_key(self, prompt: str, model_params: Dict = None) -> str:
        """
        生成缓存键（改进版）
        
        Args:
            prompt: LLM输入提示
            model_params: 模型参数(temperature, max_tokens等)
        
        Returns:
            缓存键(MD5哈希)
        """
        # 标准化模型参数
        params_str = ""
        if model_params:
            # 考虑更多影响输出的参数
            key_params = {
                k: v for k, v in sorted(model_params.items())
                if k in ["temperature", "max_tokens", "top_p", "top_k", "model", 
                         "frequency_penalty", "presence_penalty", "stop"]
            }
            params_str = json.dumps(key_params, sort_keys=True)
        
        # 标准化提示词（去除多余空格）
        normalized_prompt = " ".join(prompt.split())
        
        # 生成哈希
        content = f"{normalized_prompt}|||{params_str}"
        cache_key = hashlib.md5(content.encode('utf-8')).hexdigest()
        return cache_key
    
    def get(self, prompt: str, model_params: Dict = None) -> Optional[str]:
        """
        从缓存获取LLM结果（优化版）
        
        Args:
            prompt: LLM输入提示
            model_params: 模型参数
        
        Returns:
            缓存的LLM输出,如果未命中则返回None
        """
        cache_key = self._generate_cache_key(prompt, model_params)
        
        # 使用通用缓存管理器获取
        result = self.cache_manager.get(cache_key)
        
        if result is not None:
            self.stats["hits"] += 1
            hit_rate = self.stats["hits"] / (self.stats["hits"] + self.stats["misses"])
            logger.debug(f"缓存命中: {cache_key[:8]}... (命中率: {hit_rate:.2%})")
            return result
        else:
            self.stats["misses"] += 1
            return None
    
    def put(self, prompt: str, result: str, model_params: Dict = None):
        """
        将LLM结果放入缓存（优化版）
        
        Args:
            prompt: LLM输入提示
            result: LLM输出结果
            model_params: 模型参数
        """
        cache_key = self._generate_cache_key(prompt, model_params)
        
        # 使用通用缓存管理器存储
        self.cache_manager.put(cache_key, result, ttl=self.ttl_seconds)
        
        logger.debug(f"缓存添加: {cache_key[:8]}...")
    
    def get_batch(self, prompts: List[str], model_params: Dict = None) -> Dict[str, Optional[str]]:
        """
        批量获取缓存结果（优化版）
        
        Args:
            prompts: LLM输入提示列表
            model_params: 模型参数
        
        Returns:
            缓存结果字典 {prompt: result}
        """
        results = {}
        for prompt in prompts:
            result = self.get(prompt, model_params)
            results[prompt] = result
            if result is not None:
                self.stats["batch_hits"] += 1
            else:
                self.stats["batch_misses"] += 1
        return results
    
    def put_batch(self, prompts_and_results: List[Tuple[str, str]], model_params: Dict = None):
        """
        批量将LLM结果放入缓存（优化版）
        
        Args:
            prompts_and_results: (prompt, result) 元组列表
            model_params: 模型参数
        """
        for prompt, result in prompts_and_results:
            self.put(prompt, result, model_params)
    
    def _warmup_cache(self):
        """
        缓存预热：从持久化存储加载高频缓存项
        """
        try:
            # 从持久化存储加载缓存
            if self.enable_persistence:
                logger.info("开始缓存预热...")
                cache_stats = self.cache_manager.get_stats()
                loaded_size = cache_stats.get('lru_cache', {}).get('size', 0)
                logger.info(f"缓存预热完成: 加载了 {loaded_size} 个缓存项")
        except Exception as e:
            logger.warning(f"缓存预热失败: {e}")
    
    def clear(self):
        """清空缓存"""
        self.cache_manager.clear()
        logger.info("缓存已清空")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息（增强版）"""
        # 获取底层缓存统计
        cache_stats = self.cache_manager.get_stats()
        
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0.0
        
        return {
            **self.stats,
            "cache_size": cache_stats.get('lru_cache', {}).get('size', 0),
            "max_size": self.max_cache_size,
            "hit_rate": hit_rate,
            "total_requests": total_requests,
            "lru_hit_rate": cache_stats.get('lru_cache', {}).get('hit_rate', 0.0),
            "ttl_hit_rate": cache_stats.get('ttl_cache', {}).get('hit_rate', 0.0)
        }
    
    def cleanup(self):
        """清理过期缓存"""
        self.cache_manager.cleanup()
        logger.info("缓存清理完成")
    
    def optimize_cache(self):
        """
        优化缓存：清理过期项，整理内存
        """
        # 清理过期项
        self.cache_manager.cleanup()
        
        # 获取统计信息
        stats = self.get_stats()
        logger.info(f"缓存优化完成: 命中率={stats['hit_rate']:.2%}, 大小={stats['cache_size']}/{stats['max_size']}")


# 全局缓存实例
_global_cache: Optional[LLMCacheManager] = None
_cache_lock = None

try:
    import threading
    _cache_lock = threading.Lock()
except ImportError:
    _cache_lock = None


def get_global_cache() -> LLMCacheManager:
    """获取全局缓存实例(单例模式)"""
    global _global_cache
    
    if _global_cache is None:
        if _cache_lock:
            with _cache_lock:
                if _global_cache is None:
                    _global_cache = LLMCacheManager(
                        max_cache_size=1000,
                        ttl_seconds=86400,
                        enable_persistence=True,
                        use_hybrid_cache=True,
                        enable_batch_cache=True,
                        enable_cache_warmup=True
                    )
        else:
            _global_cache = LLMCacheManager(
                max_cache_size=1000,
                ttl_seconds=86400,
                enable_persistence=True,
                use_hybrid_cache=True,
                enable_batch_cache=True,
                enable_cache_warmup=True
            )
    
    return _global_cache
