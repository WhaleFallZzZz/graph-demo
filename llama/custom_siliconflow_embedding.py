import threading
import time
import logging
import asyncio
from typing import List, Any, Optional

from llama_index.embeddings.siliconflow import SiliconFlowEmbedding
from llama_index.core.embeddings import BaseEmbedding
import requests
from requests.exceptions import HTTPError, ConnectionError, Timeout

import httpx
import json

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
        base_url = "https://api.siliconflow.cn/v1/embeddings"
        kwargs['base_url'] = base_url
        
        super().__init__(model=model, api_key=api_key, max_retries=actual_max_retries, **kwargs)
        
        # 存储实例变量
        self._request_delay = request_delay if request_delay is not None else DEFAULT_REQUEST_DELAY
        self._max_retries = actual_max_retries
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        
        logger.info(f"CustomSiliconFlowEmbedding initialized: delay={self._request_delay}s, retries={self._max_retries}, global_concurrency=1")

    def _mean_pooling(self, embeddings: List[List[float]]) -> List[float]:
        """对多个嵌入向量进行平均池化"""
        if not embeddings:
            return []
        dim = len(embeddings[0])
        count = len(embeddings)
        avg_emb = [0.0] * dim
        for emb in embeddings:
            for i in range(dim):
                avg_emb[i] += emb[i]
        return [val / count for val in avg_emb]

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """直接调用API，不依赖父类实现，支持SSL fallback"""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self._model,
            "input": texts,
            "encoding_format": "float"
        }
        
        # 检查请求大小
        payload_json = json.dumps(payload)
        payload_size = len(payload_json)
        logger.info(f"Request payload size: {payload_size} bytes, text count: {len(texts)}")
        
        if payload_size > 1 * 1024 * 1024: # Warn > 1MB
            logger.warning(f"Large payload detected: {payload_size} bytes")

        import certifi
        
        # 定义内部函数来处理请求，方便重用
        def _do_request(verify_ssl=True):
            try:
                verify_path = certifi.where() if verify_ssl else False
                response = requests.post(
                    self._base_url,
                    headers=headers,
                    data=payload_json,
                    timeout=60,
                    verify=verify_path
                )
                response.raise_for_status()
                return response
            except requests.exceptions.SSLError as e:
                if verify_ssl:
                    logger.warning(f"SSL certificate verification failed, falling back to no verification: {e}")
                    return _do_request(verify_ssl=False)
                raise
            except requests.exceptions.ConnectionError as e:
                # 有些SSL错误会被包装在ConnectionError中
                if verify_ssl and "SSL" in str(e):
                    logger.warning(f"Connection error with SSL implications, falling back to no verification: {e}")
                    return _do_request(verify_ssl=False)
                raise

        try:
            response = _do_request(verify_ssl=True)
            data = response.json()
            
            # 解析结果
            if "data" not in data:
                raise ValueError(f"Unexpected API response format: {data}")
                
            # 按index排序确保顺序一致
            sorted_data = sorted(data["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in sorted_data]
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 413:
                logger.warning(f"Payload too large (413). Attempting to split request.")
                if len(texts) > 1:
                    # Split batch
                    mid = len(texts) // 2
                    logger.info(f"Splitting batch of {len(texts)} into {mid} and {len(texts)-mid}")
                    left = self._call_api(texts[:mid])
                    right = self._call_api(texts[mid:])
                    return left + right
                else:
                    # Single text too large
                    text = texts[0]
                    if len(text) < 100: # Too small to split
                        raise
                    
                    # Split text
                    mid = len(text) // 2
                    logger.info(f"Splitting single text of length {len(text)} into two parts")
                    part1 = text[:mid]
                    part2 = text[mid:]
                    
                    emb1 = self._call_api([part1])[0]
                    emb2 = self._call_api([part2])[0]
                    
                    # Mean pooling
                    avg_emb = self._mean_pooling([emb1, emb2])
                    return [avg_emb]
            raise

    async def _acall_api(self, texts: List[str]) -> List[List[float]]:
        """异步直接调用API，支持自动拆分过大请求和SSL fallback"""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self._model,
            "input": texts,
            "encoding_format": "float"
        }
        
        payload_json = json.dumps(payload)
        logger.info(f"Async Request payload size: {len(payload_json)} bytes, text count: {len(texts)}")

        import certifi

        async def _do_async_request(verify_ssl=True):
            try:
                verify_path = certifi.where() if verify_ssl else False
                async with httpx.AsyncClient(verify=verify_path) as client:
                    response = await client.post(
                        self._base_url,
                        headers=headers,
                        content=payload_json,
                        timeout=60
                    )
                    response.raise_for_status()
                    return response
            except httpx.ConnectError as e:
                # SSL errors often manifest as ConnectError in httpx
                if verify_ssl and ("[SSL: CERTIFICATE_VERIFY_FAILED]" in str(e) or "certificate verify failed" in str(e)):
                    logger.warning(f"Async SSL certificate verification failed, falling back to no verification: {e}")
                    return await _do_async_request(verify_ssl=False)
                raise
            except Exception as e:
                # Catch other potential SSL-related errors
                if verify_ssl and ("SSL" in str(e) or "certificate" in str(e).lower()):
                    logger.warning(f"Async SSL/Certificate error detected, falling back to no verification: {e}")
                    return await _do_async_request(verify_ssl=False)
                raise

        try:
            response = await _do_async_request(verify_ssl=True)
            data = response.json()
            
            if "data" not in data:
                raise ValueError(f"Unexpected API response format: {data}")
                
            sorted_data = sorted(data["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in sorted_data]
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 413:
                logger.warning(f"Async Payload too large (413). Attempting to split request.")
                if len(texts) > 1:
                    # Split batch
                    mid = len(texts) // 2
                    logger.info(f"Splitting async batch of {len(texts)} into {mid} and {len(texts)-mid}")
                    left = await self._acall_api(texts[:mid])
                    right = await self._acall_api(texts[mid:])
                    return left + right
                else:
                    # Single text too large
                    text = texts[0]
                    if len(text) < 100:
                        raise
                    
                    # Split text
                    mid = len(text) // 2
                    logger.info(f"Splitting single async text of length {len(text)} into two parts")
                    part1 = text[:mid]
                    part2 = text[mid:]
                    
                    emb1_list = await self._acall_api([part1])
                    emb2_list = await self._acall_api([part2])
                    
                    avg_emb = self._mean_pooling([emb1_list[0], emb2_list[0]])
                    return [avg_emb]
            raise

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

    def get_text_embedding(self, text: str) -> List[float]:
        """Override public method to ensure our logic is used"""
        return self._get_text_embedding(text)

    def get_query_embedding(self, query: str) -> List[float]:
        """Override public method to ensure our logic is used"""
        return self._get_query_embedding(query)

    async def aget_text_embedding(self, text: str) -> List[float]:
        """Override public method to ensure our logic is used"""
        return await self._aget_text_embedding(text)

    async def aget_query_embedding(self, query: str) -> List[float]:
        """Override public method to ensure our logic is used"""
        return await self._aget_query_embedding(query)

    async def aget_text_embedding_batch(
        self, texts: List[str], show_progress: bool = False, **kwargs: Any
    ) -> List[List[float]]:
        """异步批量获取文本嵌入"""
        if not texts:
            return []
            
        results = []
        BATCH_SIZE = 5
        
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            
            # 简单的重试逻辑，或者直接调用
            # 这里我们假设 _acall_api 已经处理了基本的调用
            # 但为了稳健，最好加上重试
            for attempt in range(self._max_retries):
                try:
                    if i > 0:
                        await asyncio.sleep(self._request_delay)
                        
                    batch_results = await self._acall_api(batch)
                    results.extend(batch_results)
                    break
                except Exception as e:
                    logger.error(f"异步批量嵌入失败: {e} (尝试 {attempt + 1})")
                    if attempt < self._max_retries - 1:
                        await asyncio.sleep(self._request_delay * (attempt + 1))
                        continue
                    raise
        return results

    def _get_text_embedding(self, text: str) -> List[float]:
        """获取文本嵌入，带增强的重试机制和并发控制"""
        with _GLOBAL_SYNC_SEMAPHORE:
            for attempt in range(self._max_retries):
                try:
                    # 请求前延迟
                    if attempt > 0:
                        time.sleep(self._request_delay)
                    else:
                        time.sleep(self._request_delay * 0.5)
                    
                    # 使用自定义 API 调用
                    embeddings = self._call_api([text])
                    if not embeddings:
                        raise ValueError("No embedding returned")
                    
                    result = embeddings[0]
                    
                    time.sleep(self._request_delay)
                    return result
                    
                except HTTPError as e:
                    if e.response.status_code in [403, 429]:
                        self._handle_rate_limit_error(attempt, self._max_retries, e)
                        continue
                    elif e.response.status_code == 413:
                        logger.error(f"Payload too large (413) for text length {len(text)}. Retrying with truncation might be needed.")
                        raise # 413 usually means logic error, no point retrying same payload
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
            
            raise RuntimeError(f"获取嵌入失败，重试 {self._max_retries} 次后仍失败。")

    def _get_query_embedding(self, query: str) -> List[float]:
        """获取查询嵌入"""
        return self._get_text_embedding(query)

    def get_text_embedding_batch(
        self, texts: List[str], show_progress: bool = False, **kwargs: Any
    ) -> List[List[float]]:
        """批量获取文本嵌入，带增强的重试机制和分块处理"""
        if not texts:
            return []
            
        results = []
        # 增加批处理大小，既然我们自己控制了请求
        # 之前是1，现在设为10，提高效率但保持安全
        BATCH_SIZE = 5 
        
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            
            for attempt in range(self._max_retries):
                try:
                    if i > 0 or attempt > 0:
                        time.sleep(self._request_delay)
                    
                    batch_results = self._call_api(batch)
                    results.extend(batch_results)
                    break
                    
                except Exception as e:
                    logger.error(f"批量嵌入失败: {type(e).__name__}: {e} (尝试 {attempt + 1}/{self._max_retries})")
                    if attempt < self._max_retries - 1:
                        time.sleep(self._request_delay * (attempt + 1))
                        continue
                    raise
        
        return results

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """异步获取文本嵌入"""
        import asyncio
        semaphore = _get_global_async_semaphore()
        
        async with semaphore:
            for attempt in range(self._max_retries):
                try:
                    await asyncio.sleep(self._request_delay)
                    
                    embeddings = await self._acall_api([text])
                    if not embeddings:
                        raise ValueError("No embedding returned")
                    return embeddings[0]
                    
                except Exception as e:
                    error_str = str(e)
                    if "403" in error_str or "429" in error_str:
                        wait_time = RETRY_DELAY * (2 ** attempt)
                        logger.warning(f"异步API限制错误 (403/429): {e}，等待 {wait_time:.1f} 秒后重试")
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
