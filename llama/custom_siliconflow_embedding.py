import threading
import time
import logging
import asyncio
from typing import List, Any, Optional

from llama_index.embeddings.siliconflow import SiliconFlowEmbedding
from llama_index.core.embeddings import BaseEmbedding
import requests
from requests.exceptions import HTTPError, ConnectionError, Timeout

logger = logging.getLogger(__name__)

# 全局信号量，确保跨实例的全局并发控制
_GLOBAL_SYNC_SEMAPHORE = threading.Semaphore(1)
_GLOBAL_ASYNC_SEMAPHORE = None

def _get_global_async_semaphore():
    """延迟初始化全局异步信号量"""
    global _GLOBAL_ASYNC_SEMAPHORE
    import asyncio
    if _GLOBAL_ASYNC_SEMAPHORE is None:
        _GLOBAL_ASYNC_SEMAPHORE = asyncio.Semaphore(1)
    return _GLOBAL_ASYNC_SEMAPHORE

# 从配置文件导入速率限制配置 - 优化在线API性能
try:
    from config import RATE_LIMIT_CONFIG, EMBEDDING_CONFIG
    DEFAULT_REQUEST_DELAY = RATE_LIMIT_CONFIG.get('request_delay', 0.5)  # 减少延迟到0.5秒
    MAX_RETRIES = RATE_LIMIT_CONFIG.get('max_retries', 2)  # 减少重试次数
    RETRY_DELAY = RATE_LIMIT_CONFIG.get('retry_delay', 5.0)  # 减少重试延迟
    ENABLE_LOCAL_FALLBACK = False  # 强制禁用本地回退
    USE_HYBRID_EMBEDDING = False  # 强制禁用混合嵌入
    MODEL_KWARGS = {}  # 空配置，不加载本地模型
except ImportError:
    DEFAULT_REQUEST_DELAY = 0.5  # 默认减少延迟
    MAX_RETRIES = 2  # 默认减少重试
    RETRY_DELAY = 5.0
    ENABLE_LOCAL_FALLBACK = False  # 默认禁用回退
    USE_HYBRID_EMBEDDING = False  # 默认只用在线
    MODEL_KWARGS = {}

class HybridSiliconFlowEmbedding(BaseEmbedding):
    """
    轻量级在线嵌入模型：只用SiliconFlow API，不加载本地模型
    继承自BaseEmbedding以确保兼容性，优化CPU和内存占用
    """
    
    def __init__(
        self,
        model: str = "BAAI/bge-m3",
        api_key: str = None,
        max_retries: int = None,
        request_delay: float = None,
        **kwargs: Any,
    ) -> None:
        # 初始化BaseEmbedding
        super().__init__(**kwargs)
        
        # 使用私有变量避免Pydantic字段冲突 - 轻量级配置
        self._model_name = model
        self._api_key = api_key
        self._enable_local_fallback = False  # 强制禁用本地回退
        self._max_retries = max_retries if max_retries is not None else MAX_RETRIES
        self._request_delay = request_delay if request_delay is not None else DEFAULT_REQUEST_DELAY
        
        # 只初始化SiliconFlow客户端
        self._siliconflow_embedding = None
        self._siliconflow_available = False
        self._local_embedding = None  # 始终为None
        
        self._initialize_models()

    def _initialize_models(self):
        """初始化嵌入模型 - 只用在线API"""
        # 只用在线SiliconFlow API，不加载本地模型
        logger.info("只用在线SiliconFlow API，不加载本地模型")
        
        # 初始化SiliconFlow客户端
        try:
            self._siliconflow_embedding = CustomSiliconFlowEmbedding(
                model=self._model_name,
                api_key=self._api_key,
                max_retries=self._max_retries,
                request_delay=self._request_delay
            )
            self._siliconflow_available = True
            logger.info("SiliconFlow API 初始化成功 - 只用在线模式")
        except Exception as e:
            self._siliconflow_available = False
            logger.error(f"SiliconFlow API 初始化失败: {e}")
            raise RuntimeError("SiliconFlow API 初始化失败，且本地回退被禁用")
        
        # 不初始化本地模型，节省内存和CPU
        self._local_embedding = None

    def _get_text_embedding_with_fallback(self, text: str) -> List[float]:
        """获取文本嵌入 - 只用在线API"""
        
        # 只用SiliconFlow API，不尝试本地模型
        if self._siliconflow_available and self._siliconflow_embedding:
            try:
                logger.debug(f"使用SiliconFlow API获取嵌入: '{text[:50]}...'")
                result = self._siliconflow_embedding._get_text_embedding(text)
                logger.debug("SiliconFlow API 成功")
                return result
            except (HTTPError, ConnectionError, Timeout) as api_error:
                logger.error(f"SiliconFlow API 失败: {api_error}")
                raise RuntimeError(f"SiliconFlow API 获取嵌入失败: {api_error}")
            except Exception as e:
                logger.error(f"SiliconFlow API 错误: {e}")
                raise RuntimeError(f"SiliconFlow API 获取嵌入失败: {e}")
        
        # SiliconFlow不可用
        raise RuntimeError("SiliconFlow API 不可用，本地模型已禁用")

    def _get_text_embedding(self, text: str) -> List[float]:
        """获取文本嵌入的主入口"""
        return self._get_text_embedding_with_fallback(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        """获取查询嵌入"""
        return self._get_text_embedding_with_fallback(query)

    def get_text_embedding_batch(
        self, texts: List[str], show_progress: bool = False, **kwargs: Any
    ) -> List[List[float]]:
        """批量获取文本嵌入，逐个处理以避免触发限制"""
        results = []
        
        for i, text in enumerate(texts):
            try:
                embedding = self._get_text_embedding(text)
                results.append(embedding)
                
                # 批量处理时的额外延迟
                if i < len(texts) - 1:
                    time.sleep(self._request_delay)
                    
            except Exception as e:
                logger.error(f"批量处理中第{i+1}个文本嵌入失败: {e}")
                # 可以选择跳过失败的文本或终止整个批次
                # 这里选择终止整个批次
                raise RuntimeError(f"批量嵌入失败于第{i+1}个文本: {e}")
        
        return results

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """异步获取文本嵌入 - 只用在线API"""
        
        # 只用SiliconFlow API，不尝试本地模型
        if self._siliconflow_available and self._siliconflow_embedding:
            try:
                await asyncio.sleep(self._request_delay)
                result = await self._siliconflow_embedding._aget_text_embedding(text)
                return result
            except Exception as e:
                logger.error(f"异步SiliconFlow API失败: {e}")
                raise RuntimeError(f"异步SiliconFlow API获取嵌入失败: {e}")
        
        # SiliconFlow不可用
        raise RuntimeError("异步SiliconFlow API 不可用，本地模型已禁用")

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """异步获取查询嵌入"""
        return await self._aget_text_embedding(query)


class CustomSiliconFlowEmbedding(SiliconFlowEmbedding):
    """
    原有的CustomSiliconFlowEmbedding类，保持兼容性
    """
    
    def __init__(
        self,
        model: str,
        api_key: str,
        max_retries: int = None,
        request_delay: float = None,
        **kwargs: Any,
    ) -> None:
        # 从kwargs中移除配置参数，避免传递给父类
        kwargs.pop('request_delay', None)
        kwargs.pop('max_retries', None)
        
        # 设置重试参数 - 使用默认值
        actual_max_retries = max_retries if max_retries is not None else MAX_RETRIES
        
        # 设置正确的SiliconFlow API基础URL
        kwargs['base_url'] = "https://api.siliconflow.cn/v1/embeddings"
        
        super().__init__(model=model, api_key=api_key, max_retries=actual_max_retries, **kwargs)
        
        # 存储实例变量
        self._request_delay = request_delay if request_delay is not None else DEFAULT_REQUEST_DELAY
        self._max_retries = actual_max_retries
        
        logger.info(f"CustomSiliconFlowEmbedding initialized: delay={self._request_delay}s, retries={self._max_retries}, global_concurrency=1")

    def _handle_rate_limit_error(self, attempt: int, max_attempts: int, error: Exception) -> float:
        """处理速率限制错误，返回等待时间"""
        # 指数退避 + 随机抖动
        base_delay = RETRY_DELAY * (2 ** attempt)
        jitter = 1.0  # 固定抖动值
        wait_time = base_delay + jitter
        
        if attempt < max_attempts - 1:
            logger.warning(f"API限制错误 (403/429)，等待 {wait_time:.1f} 秒后重试 (尝试 {attempt + 1}/{max_attempts})")
            logger.warning(f"错误详情: {error}")
            time.sleep(wait_time)
            return wait_time
        else:
            logger.error(f"API限制错误，已达到最大重试次数 {max_attempts}")
            return 0

    def _get_text_embedding(self, text: str) -> List[float]:
        """获取文本嵌入，带增强的重试机制和并发控制"""
        with _GLOBAL_SYNC_SEMAPHORE:
            for attempt in range(self._max_retries):
                try:
                    # 请求前延迟
                    if attempt > 0:  # 第一次尝试不需要额外延迟
                        time.sleep(self._request_delay)
                    else:
                        # 即使是第一次，为了安全也稍微延迟一点点，避免连续调用太快
                        time.sleep(self._request_delay * 0.5)
                    
                    # 添加请求头信息
                    result = super()._get_text_embedding(text)
                    
                    # 成功后添加额外延迟，避免触发速率限制
                    time.sleep(self._request_delay)
                    return result
                    
                except HTTPError as e:
                    if e.response.status_code in [403, 429]:  # Forbidden or Too Many Requests
                        self._handle_rate_limit_error(attempt, self._max_retries, e)
                        continue
                    else:
                        logger.error(f"HTTP错误 {e.response.status_code}: {e}")
                        if attempt < self._max_retries - 1:
                            time.sleep(self._request_delay)
                            continue
                        raise
                        
                except (ConnectionError, Timeout) as e:
                    logger.warning(f"连接错误: {e}，尝试重连 (尝试 {attempt + 1}/{self._max_retries})")
                    if attempt < self._max_retries - 1:
                        time.sleep(self._request_delay * (attempt + 1))
                        continue
                    raise
                    
                except Exception as e:
                    logger.error(f"嵌入失败: {type(e).__name__}: {e} (尝试 {attempt + 1}/{self._max_retries})")
                    if attempt < self._max_retries - 1:
                        time.sleep(self._request_delay * (attempt + 1))
                        continue
                    raise
            
            raise RuntimeError(f"获取嵌入失败，重试 {self._max_retries} 次后仍失败。请检查API密钥和速率限制。")

    def _get_query_embedding(self, query: str) -> List[float]:
        """获取查询嵌入，带增强的重试机制"""
        # 复用文本嵌入的逻辑
        return self._get_text_embedding(query)

    def get_text_embedding_batch(
        self, texts: List[str], show_progress: bool = False, **kwargs: Any
    ) -> List[List[float]]:
        """批量获取文本嵌入，带增强的重试机制和更保守的速率控制"""
        if not texts:
            return []
            
        results = []
        batch_size = 1  # 严格控制批量大小，避免触发速率限制
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            for attempt in range(self._max_retries):
                try:
                    # 批次前的延迟
                    if i > 0 or attempt > 0:  # 第一个批次第一次尝试不需要延迟
                        time.sleep(self._request_delay * 2)  # 批次间使用更长的延迟
                    
                    # 处理单个批次
                    batch_results = []
                    for text in batch:
                        # 复用 _get_text_embedding，它已经包含了信号量锁
                        embedding = self._get_text_embedding(text)
                        batch_results.append(embedding)
                    
                    results.extend(batch_results)
                    break  # 成功处理批次，跳出重试循环
                    
                except Exception as e:
                    logger.error(f"批量嵌入失败: {type(e).__name__}: {e} (尝试 {attempt + 1}/{self._max_retries})")
                    if attempt < self._max_retries - 1:
                        time.sleep(self._request_delay * (attempt + 1))
                        continue
                    raise
        
        return results

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """异步获取文本嵌入，带增强的重试机制和并发控制"""
        import asyncio
        semaphore = _get_global_async_semaphore()
        
        async with semaphore:
            for attempt in range(self._max_retries):
                try:
                    # 每次请求前都等待，即使是第一次，确保完全串行且有间隔
                    await asyncio.sleep(self._request_delay)
                    
                    result = await super()._aget_text_embedding(text)
                    return result
                    
                except Exception as e:
                    error_str = str(e)
                    # 检查是否为速率限制或权限错误 (403/429)
                    if "403" in error_str or "429" in error_str:
                        # 使用较长的重试延迟
                        wait_time = RETRY_DELAY * (2 ** attempt)
                        logger.warning(f"异步API限制错误 (403/429): {e}，等待 {wait_time:.1f} 秒后重试 (尝试 {attempt + 1}/{self._max_retries})")
                        if attempt < self._max_retries - 1:
                            await asyncio.sleep(wait_time)
                            continue
                    
                    logger.error(f"异步嵌入失败: {e} (尝试 {attempt + 1}/{self._max_retries})")
                    if attempt < self._max_retries - 1:
                        await asyncio.sleep(self._request_delay * (attempt + 1))
                        continue
                    raise
            
            raise RuntimeError(f"异步获取嵌入失败，重试 {self._max_retries} 次后仍失败")

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """异步获取查询嵌入"""
        return await self._aget_text_embedding(query)