"""
自定义 SiliconFlow 嵌入模型
提供增强的嵌入功能，包括：
- 在线 API 调用（SiliconFlow）
- 自动重试机制
- 速率限制处理
- SSL 证书验证回退
- 请求自动拆分（处理大文本）
- 全局并发控制
- 同步和异步支持
"""

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
# 防止多个实例同时发起过多请求导致 API 限流
_GLOBAL_SYNC_SEMAPHORE = threading.Semaphore(1)
_GLOBAL_ASYNC_SEMAPHORE = None


def _get_global_async_semaphore():
    """
    延迟初始化全局异步信号量
    
    Returns:
        asyncio.Semaphore: 全局异步信号量实例
    """
    global _GLOBAL_ASYNC_SEMAPHORE
    import asyncio
    if _GLOBAL_ASYNC_SEMAPHORE is None:
        _GLOBAL_ASYNC_SEMAPHORE = asyncio.Semaphore(1)
    return _GLOBAL_ASYNC_SEMAPHORE


# 从配置文件导入速率限制配置 - 优化在线 API 性能
try:
    from config import RATE_LIMIT_CONFIG, EMBEDDING_CONFIG
    DEFAULT_REQUEST_DELAY = RATE_LIMIT_CONFIG.get('request_delay', 0.5)
    MAX_RETRIES = RATE_LIMIT_CONFIG.get('max_retries', 2)
    RETRY_DELAY = RATE_LIMIT_CONFIG.get('retry_delay', 5.0)
    ENABLE_LOCAL_FALLBACK = False
    USE_HYBRID_EMBEDDING = False
    MODEL_KWARGS = {}
except ImportError:
    DEFAULT_REQUEST_DELAY = 0.5
    MAX_RETRIES = 2
    RETRY_DELAY = 5.0
    ENABLE_LOCAL_FALLBACK = False
    USE_HYBRID_EMBEDDING = False
    MODEL_KWARGS = {}


class HybridSiliconFlowEmbedding(BaseEmbedding):
    """
    轻量级在线嵌入模型
    
    特点：
    - 只使用 SiliconFlow API，不加载本地模型
    - 继承自 BaseEmbedding 以确保兼容性
    - 优化 CPU 和内存占用
    
    适用场景：
    - 需要快速初始化的嵌入服务
    - 不需要本地回退的场景
    - 资源受限的环境
    """
    
    def __init__(
        self,
        model: str = "BAAI/bge-m3",
        api_key: str = None,
        max_retries: int = None,
        request_delay: float = None,
        **kwargs: Any,
    ) -> None:
        """
        初始化轻量级在线嵌入模型
        
        Args:
            model: 模型名称，默认为 BAAI/bge-m3
            api_key: SiliconFlow API 密钥
            max_retries: 最大重试次数，默认使用配置文件中的值
            request_delay: 请求延迟（秒），默认使用配置文件中的值
            **kwargs: 其他传递给 BaseEmbedding 的参数
        """
        super().__init__(**kwargs)
        
        self._model_name = model
        self._api_key = api_key
        self._enable_local_fallback = False
        self._max_retries = max_retries if max_retries is not None else MAX_RETRIES
        self._request_delay = request_delay if request_delay is not None else DEFAULT_REQUEST_DELAY
        
        self._siliconflow_embedding = None
        self._siliconflow_available = False
        self._local_embedding = None
        
        self._initialize_models()

    def _initialize_models(self):
        """
        初始化嵌入模型
        
        只使用在线 SiliconFlow API，不加载本地模型以节省资源
        """
        logger.debug("初始化在线 SiliconFlow API，不加载本地模型")
        
        try:
            self._siliconflow_embedding = CustomSiliconFlowEmbedding(
                model=self._model_name,
                api_key=self._api_key,
                max_retries=self._max_retries,
                request_delay=self._request_delay
            )
            self._siliconflow_available = True
            logger.debug("SiliconFlow API 初始化成功 - 在线模式")
        except Exception as e:
            self._siliconflow_available = False
            logger.error(f"SiliconFlow API 初始化失败: {e}")
            raise RuntimeError("SiliconFlow API 初始化失败，且本地回退被禁用")
        
        self._local_embedding = None

    def _get_text_embedding_with_fallback(self, text: str) -> List[float]:
        """
        获取文本嵌入（仅在线 API）
        
        Args:
            text: 输入文本
            
        Returns:
            List[float]: 嵌入向量
            
        Raises:
            RuntimeError: 当 API 不可用时抛出异常
        """
        if self._siliconflow_available and self._siliconflow_embedding:
            try:
                logger.debug(f"使用 SiliconFlow API 获取嵌入: '{text[:50]}...'")
                result = self._siliconflow_embedding._get_text_embedding(text)
                logger.debug("SiliconFlow API 调用成功")
                return result
            except (HTTPError, ConnectionError, Timeout) as api_error:
                logger.error(f"SiliconFlow API 调用失败: {api_error}")
                raise RuntimeError(f"SiliconFlow API 获取嵌入失败: {api_error}")
            except Exception as e:
                logger.error(f"SiliconFlow API 错误: {e}")
                raise RuntimeError(f"SiliconFlow API 获取嵌入失败: {e}")
        
        raise RuntimeError("SiliconFlow API 不可用，本地模型已禁用")

    def _get_text_embedding(self, text: str) -> List[float]:
        """
        获取文本嵌入的主入口
        
        Args:
            text: 输入文本
            
        Returns:
            List[float]: 嵌入向量
        """
        return self._get_text_embedding_with_fallback(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        """
        获取查询嵌入
        
        Args:
            query: 查询文本
            
        Returns:
            List[float]: 嵌入向量
        """
        return self._get_text_embedding_with_fallback(query)

    def get_text_embedding_batch(
        self, texts: List[str], show_progress: bool = False, **kwargs: Any
    ) -> List[List[float]]:
        """
        批量获取文本嵌入
        
        逐个处理以避免触发 API 限制
        
        Args:
            texts: 文本列表
            show_progress: 是否显示进度（未使用）
            **kwargs: 其他参数
            
        Returns:
            List[List[float]]: 嵌入向量列表
        """
        results = []
        
        for i, text in enumerate(texts):
            try:
                embedding = self._get_text_embedding(text)
                results.append(embedding)
                
                if i < len(texts) - 1:
                    time.sleep(self._request_delay)
                    
            except Exception as e:
                logger.error(f"批量处理中第 {i+1} 个文本嵌入失败: {e}")
                raise RuntimeError(f"批量嵌入失败于第 {i+1} 个文本: {e}")
        
        return results

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """
        异步获取文本嵌入（仅在线 API）
        
        Args:
            text: 输入文本
            
        Returns:
            List[float]: 嵌入向量
            
        Raises:
            RuntimeError: 当 API 不可用时抛出异常
        """
        if self._siliconflow_available and self._siliconflow_embedding:
            try:
                await asyncio.sleep(self._request_delay)
                result = await self._siliconflow_embedding._aget_text_embedding(text)
                return result
            except Exception as e:
                logger.error(f"异步 SiliconFlow API 失败: {e}")
                raise RuntimeError(f"异步 SiliconFlow API 获取嵌入失败: {e}")
        
        raise RuntimeError("异步 SiliconFlow API 不可用，本地模型已禁用")

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """
        异步获取查询嵌入
        
        Args:
            query: 查询文本
            
        Returns:
            List[float]: 嵌入向量
        """
        return await self._aget_text_embedding(query)


class CustomSiliconFlowEmbedding(SiliconFlowEmbedding):
    """
    自定义 SiliconFlow 嵌入模型
    
    特点：
    - 继承自 SiliconFlowEmbedding，保持兼容性
    - 增强的重试机制
    - 自动处理 SSL 证书验证失败
    - 自动拆分过大的请求
    - 全局并发控制
    - 支持同步和异步调用
    
    适用场景：
    - 需要稳定可靠的嵌入服务
    - 需要处理网络不稳定的情况
    - 需要处理大文本的场景
    """
    
    def __init__(
        self,
        model: str,
        api_key: str,
        max_retries: int = None,
        request_delay: float = None,
        **kwargs: Any,
    ) -> None:
        """
        初始化自定义 SiliconFlow 嵌入模型
        
        Args:
            model: 模型名称
            api_key: SiliconFlow API 密钥
            max_retries: 最大重试次数
            request_delay: 请求延迟（秒）
            **kwargs: 其他参数
        """
        kwargs.pop('request_delay', None)
        kwargs.pop('max_retries', None)
        
        actual_max_retries = max_retries if max_retries is not None else MAX_RETRIES
        
        base_url = "https://api.siliconflow.cn/v1/embeddings"
        kwargs['base_url'] = base_url
        
        super().__init__(model=model, api_key=api_key, max_retries=actual_max_retries, **kwargs)
        
        self._request_delay = request_delay if request_delay is not None else DEFAULT_REQUEST_DELAY
        self._max_retries = actual_max_retries
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        
        logger.debug(
            f"CustomSiliconFlowEmbedding 初始化完成: "
            f"delay={self._request_delay}s, "
            f"retries={self._max_retries}, "
            f"global_concurrency=1"
        )

    def _mean_pooling(self, embeddings: List[List[float]]) -> List[float]:
        """
        对多个嵌入向量进行平均池化
        
        Args:
            embeddings: 嵌入向量列表
            
        Returns:
            List[float]: 平均后的嵌入向量
        """
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
        """
        直接调用 API，支持 SSL fallback 和自动拆分
        
        Args:
            texts: 文本列表
            
        Returns:
            List[List[float]]: 嵌入向量列表
            
        Raises:
            ValueError: API 响应格式错误
            HTTPError: HTTP 请求错误
        """
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
        payload_size = len(payload_json)
        logger.debug(f"请求负载大小: {payload_size} bytes, 文本数量: {len(texts)}")
        
        if payload_size > 1 * 1024 * 1024:
            logger.warning(f"检测到大负载: {payload_size} bytes")

        import certifi
        
        def _do_request(verify_ssl=True):
            """
            执行 HTTP 请求
            
            Args:
                verify_ssl: 是否验证 SSL 证书
                
            Returns:
                requests.Response: HTTP 响应对象
            """
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
                    logger.debug(f"SSL 证书验证失败，回退到不验证模式: {e}")
                    return _do_request(verify_ssl=False)
                raise
            except requests.exceptions.ConnectionError as e:
                if verify_ssl and "SSL" in str(e):
                    logger.debug(f"连接错误包含 SSL 问题，回退到不验证模式: {e}")
                    return _do_request(verify_ssl=False)
                raise

        try:
            response = _do_request(verify_ssl=True)
            data = response.json()
            
            if "data" not in data:
                raise ValueError(f"意外的 API 响应格式: {data}")
                
            sorted_data = sorted(data["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in sorted_data]
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 413:
                logger.warning(f"负载过大 (413)。尝试拆分请求。")
                if len(texts) > 1:
                    mid = len(texts) // 2
                    logger.debug(f"拆分批次 {len(texts)} 为 {mid} 和 {len(texts)-mid}")
                    left = self._call_api(texts[:mid])
                    right = self._call_api(texts[mid:])
                    return left + right
                else:
                    text = texts[0]
                    if len(text) < 100:
                        raise
                    
                    mid = len(text) // 2
                    logger.debug(f"拆分单个文本 (长度 {len(text)}) 为两部分")
                    part1 = text[:mid]
                    part2 = text[mid:]
                    
                    emb1 = self._call_api([part1])[0]
                    emb2 = self._call_api([part2])[0]
                    
                    avg_emb = self._mean_pooling([emb1, emb2])
                    return [avg_emb]
            raise

    async def _acall_api(self, texts: List[str]) -> List[List[float]]:
        """
        异步直接调用 API，支持自动拆分过大请求和 SSL fallback
        
        Args:
            texts: 文本列表
            
        Returns:
            List[List[float]]: 嵌入向量列表
            
        Raises:
            ValueError: API 响应格式错误
            httpx.HTTPStatusError: HTTP 状态错误
        """
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
        logger.debug(f"异步请求负载大小: {len(payload_json)} bytes, 文本数量: {len(texts)}")

        import certifi

        async def _do_async_request(verify_ssl=True):
            """
            执行异步 HTTP 请求
            
            Args:
                verify_ssl: 是否验证 SSL 证书
                
            Returns:
                httpx.Response: HTTP 响应对象
            """
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
                if verify_ssl and ("[SSL: CERTIFICATE_VERIFY_FAILED]" in str(e) or "certificate verify failed" in str(e)):
                    logger.debug(f"异步 SSL 证书验证失败，回退到不验证模式: {e}")
                    return await _do_async_request(verify_ssl=False)
                raise
            except Exception as e:
                if verify_ssl and ("SSL" in str(e) or "certificate" in str(e).lower()):
                    logger.debug(f"检测到异步 SSL/证书错误，回退到不验证模式: {e}")
                    return await _do_async_request(verify_ssl=False)
                raise

        try:
            response = await _do_async_request(verify_ssl=True)
            data = response.json()
            
            if "data" not in data:
                raise ValueError(f"意外的 API 响应格式: {data}")
                
            sorted_data = sorted(data["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in sorted_data]
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 413:
                logger.warning(f"异步负载过大 (413)。尝试拆分请求。")
                if len(texts) > 1:
                    mid = len(texts) // 2
                    logger.debug(f"拆分异步批次 {len(texts)} 为 {mid} 和 {len(texts)-mid}")
                    left = await self._acall_api(texts[:mid])
                    right = await self._acall_api(texts[mid:])
                    return left + right
                else:
                    text = texts[0]
                    if len(text) < 100:
                        raise
                    
                    mid = len(text) // 2
                    logger.debug(f"拆分单个异步文本 (长度 {len(text)}) 为两部分")
                    part1 = text[:mid]
                    part2 = text[mid:]
                    
                    emb1_list = await self._acall_api([part1])
                    emb2_list = await self._acall_api([part2])
                    
                    avg_emb = self._mean_pooling([emb1_list[0], emb2_list[0]])
                    return [avg_emb]
            raise

    def _handle_rate_limit_error(self, attempt: int, max_attempts: int, error: Exception) -> float:
        """
        处理速率限制错误，返回等待时间
        
        使用指数退避策略
        
        Args:
            attempt: 当前尝试次数
            max_attempts: 最大尝试次数
            error: 错误对象
            
        Returns:
            float: 等待时间（秒）
        """
        base_delay = RETRY_DELAY * (2 ** attempt)
        jitter = 1.0
        wait_time = base_delay + jitter
        
        if attempt < max_attempts - 1:
            logger.warning(
                f"API 限制错误 (403/429)，等待 {wait_time:.1f} 秒后重试 "
                f"(尝试 {attempt + 1}/{max_attempts})"
            )
            logger.warning(f"错误详情: {error}")
            time.sleep(wait_time)
            return wait_time
        else:
            logger.error(f"API 限制错误，已达到最大重试次数 {max_attempts}")
            return 0

    def get_text_embedding(self, text: str) -> List[float]:
        """
        获取文本嵌入（公共方法）
        
        Args:
            text: 输入文本
            
        Returns:
            List[float]: 嵌入向量
        """
        return self._get_text_embedding(text)

    def get_query_embedding(self, query: str) -> List[float]:
        """
        获取查询嵌入（公共方法）
        
        Args:
            query: 查询文本
            
        Returns:
            List[float]: 嵌入向量
        """
        return self._get_query_embedding(query)

    async def aget_text_embedding(self, text: str) -> List[float]:
        """
        异步获取文本嵌入（公共方法）
        
        Args:
            text: 输入文本
            
        Returns:
            List[float]: 嵌入向量
        """
        return await self._aget_text_embedding(text)

    async def aget_query_embedding(self, query: str) -> List[float]:
        """
        异步获取查询嵌入（公共方法）
        
        Args:
            query: 查询文本
            
        Returns:
            List[float]: 嵌入向量
        """
        return await self._aget_query_embedding(query)

    async def aget_text_embedding_batch(
        self, texts: List[str], show_progress: bool = False, **kwargs: Any
    ) -> List[List[float]]:
        """
        异步批量获取文本嵌入
        
        使用线程池并发执行同步 API 调用
        
        Args:
            texts: 文本列表
            show_progress: 是否显示进度（未使用）
            **kwargs: 其他参数
            
        Returns:
            List[List[float]]: 嵌入向量列表
        """
        if not texts:
            return []
        import os
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        max_workers = max(2, (os.cpu_count() or 4) // 2)
        batch_size = 8
        loop = asyncio.get_event_loop()
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            tasks = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                tasks.append(loop.run_in_executor(executor, self._call_api, batch))
            done = await asyncio.gather(*tasks, return_exceptions=True)
            for item in done:
                if isinstance(item, Exception):
                    raise item
                results.extend(item)
        return results

    def _get_text_embedding(self, text: str) -> List[float]:
        """
        获取文本嵌入，带增强的重试机制和并发控制
        
        Args:
            text: 输入文本
            
        Returns:
            List[float]: 嵌入向量
            
        Raises:
            RuntimeError: 重试失败后抛出异常
        """
        with _GLOBAL_SYNC_SEMAPHORE:
            for attempt in range(self._max_retries):
                try:
                    if attempt > 0:
                        time.sleep(self._request_delay)
                    else:
                        time.sleep(self._request_delay * 0.5)
                    
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
                        logger.error(f"负载过大 (413) 文本长度 {len(text)}。可能需要截断重试。")
                        raise
                    else:
                        logger.error(f"HTTP 错误 {e.response.status_code}: {e}")
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
        """
        获取查询嵌入
        
        Args:
            query: 查询文本
            
        Returns:
            List[float]: 嵌入向量
        """
        return self._get_text_embedding(query)

    def get_text_embedding_batch(
        self, texts: List[str], show_progress: bool = False, **kwargs: Any
    ) -> List[List[float]]:
        """
        批量获取文本嵌入，带增强的重试机制和分块处理
        
        Args:
            texts: 文本列表
            show_progress: 是否显示进度（未使用）
            **kwargs: 其他参数
            
        Returns:
            List[List[float]]: 嵌入向量列表
        """
        if not texts:
            return []
            
        results = []
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
        """
        异步获取文本嵌入
        
        Args:
            text: 输入文本
            
        Returns:
            List[float]: 嵌入向量
            
        Raises:
            RuntimeError: 重试失败后抛出异常
        """
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
                        logger.warning(f"异步 API 限制错误 (403/429): {e}，等待 {wait_time:.1f} 秒后重试")
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
        """
        异步获取查询嵌入
        
        Args:
            query: 查询文本
            
        Returns:
            List[float]: 嵌入向量
        """
        return await self._aget_text_embedding(query)
