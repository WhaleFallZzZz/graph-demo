"""
自定义 SiliconFlow Rerank 重排序器

提供基于 SiliconFlow API 的文档重排序功能，用于优化检索结果的相关性排序。

主要功能：
- 使用 SiliconFlow 的 Rerank API 进行文档重排序
- 支持自定义模型和返回节点数量
- 自动降级机制：API 失败时返回原始节点
- 集成到 LlamaIndex 的后处理器框架

使用场景：
- 检索增强生成（RAG）中的结果重排序
- 知识图谱查询中的节点相关性排序
- 实体提取中的过滤和排序
"""

from typing import List, Optional, Any
import requests
import logging
import certifi
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
try:
    from llama.config import get_rate_limit
except ImportError:
    try:
        from config import get_rate_limit
    except ImportError:
        get_rate_limit = None

logger = logging.getLogger(__name__)


class CustomSiliconFlowRerank(BaseNodePostprocessor):
    """
    SiliconFlow 重排序器
    
    使用 SiliconFlow 的 Rerank API 对检索到的节点进行重排序，
    根据查询的相关性对节点重新评分和排序。
    
    特点：
    - 基于 BAAI/bge-reranker-v2-m3 等先进重排序模型
    - 支持自定义返回节点数量（top_n）
    - API 失败时自动降级为原始排序
    - 集成到 LlamaIndex 后处理器框架
    
    适用场景：
    - RAG 系统中的检索结果优化
    - 知识图谱查询中的节点排序
    - 需要精确相关性排序的应用
    """
    
    api_key: str = Field(description="SiliconFlow API 密钥")
    model: str = Field(description="Rerank 模型名称")
    top_n: int = Field(default=2, description="返回的节点数量")
    base_url: str = Field(default="https://api.siliconflow.cn/v1/rerank", description="API 基础 URL")
    
    def __init__(
        self,
        api_key: str,
        model: str = "BAAI/bge-reranker-v2-m3",
        top_n: int = 2,
        base_url: str = "https://api.siliconflow.cn/v1/rerank",
        request_delay: Optional[float] = None,
        max_retries: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """
        初始化 SiliconFlow 重排序器
        
        Args:
            api_key: SiliconFlow API 密钥
            model: Rerank 模型名称，默认为 BAAI/bge-reranker-v2-m3
            top_n: 返回的节点数量，默认为 2
            base_url: API 基础 URL，默认为 https://api.siliconflow.cn/v1/rerank
            request_delay: 请求延迟，若不提供则从 config 获取
            max_retries: 最大重试次数，若不提供则从 config 获取
            **kwargs: 其他传递给父类的参数
        """
        super().__init__(
            api_key=api_key,
            model=model,
            top_n=top_n,
            base_url=base_url,
            **kwargs
        )
        
        # 获取频控配置
        if get_rate_limit:
            limit_info = get_rate_limit(model)
            self._request_delay = request_delay if request_delay is not None else limit_info["request_delay"]
            self._max_retries = max_retries if max_retries is not None else limit_info["max_retries"]
            self._retry_delay = limit_info["retry_delay"]
        else:
            self._request_delay = request_delay if request_delay is not None else 0.5
            self._max_retries = max_retries if max_retries is not None else 3
            self._retry_delay = 5.0
            
        logger.debug(f"CustomSiliconFlowRerank 初始化: model={model}, delay={self._request_delay:.4f}s")

    @classmethod
    def class_name(cls) -> str:
        """
        返回类名称
        
        Returns:
            str: 类名称 "CustomSiliconFlowRerank"
        """
        return "CustomSiliconFlowRerank"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """
        对节点进行后处理（重排序）
        
        使用 SiliconFlow Rerank API 对节点进行重排序，
        根据查询的相关性重新计算节点分数。
        
        Args:
            nodes: 待重排序的节点列表
            query_bundle: 查询包，包含查询字符串
            
        Returns:
            List[NodeWithScore]: 重排序后的节点列表
            
        Raises:
            ValueError: 当 query_bundle 为 None 时抛出异常
            
        降级机制：
            如果 API 调用失败，返回原始节点的前 top_n 个节点
        """
        if not nodes:
            return []
            
        if query_bundle is None:
            raise ValueError("Query bundle must be provided.")
            
        query_str = query_bundle.query_str
        
        logger.debug(f"开始重排序 {len(nodes)} 个节点，查询: '{query_str[:50]}...'")
        
        # 准备请求数据
        documents = [node.node.get_content() for node in nodes]
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "query": query_str,
            "documents": documents,
            "top_n": self.top_n,
            "return_documents": False 
        }
        
        def _do_request(verify_ssl=True):
            """
            执行 HTTP 请求，支持 SSL 证书验证回退
            
            Args:
                verify_ssl: 是否验证 SSL 证书
                
            Returns:
                requests.Response: HTTP 响应对象
            """
            try:
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=30,
                    verify=False
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
        
        import time
        max_retries = self._max_retries
        retry_delay = self._retry_delay
        result = None
        
        # 在请求前执行延迟
        time.sleep(self._request_delay)
        
        for attempt in range(max_retries):
            try:
                response = _do_request(verify_ssl=True)
                result = response.json()
                break
            except (requests.exceptions.HTTPError, requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                status_code = getattr(getattr(e, 'response', None), 'status_code', None)
                if status_code in [403, 429, 500, 502, 503, 504] or isinstance(e, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        logger.warning(f"Rerank API 错误 ({status_code or 'N/A'}): {e}，将在 {wait_time:.1f} 秒后重试 (第 {attempt + 1} 次)")
                        time.sleep(wait_time)
                        continue
                
                logger.error(f"Rerank API 最终失败: {e}")
                return nodes[:self.top_n]
            except Exception as e:
                logger.error(f"Rerank 未知错误: {e}")
                return nodes[:self.top_n]
        
        if not result:
            return nodes[:self.top_n]

        # 解析结果
        # 预期格式: {"results": [{"index": 0, "relevance_score": 0.9}, ...]}
        rerank_results = result.get("results", [])
        
        if not rerank_results:
            logger.warning("Rerank API 返回空结果，使用原始排序")
            return nodes[:self.top_n]
        
        new_nodes = []
        for item in rerank_results:
            idx = item.get("index")
            score = item.get("relevance_score")
            
            if idx is not None and 0 <= idx < len(nodes):
                node = nodes[idx]
                node.score = score
                new_nodes.append(node)
                logger.debug(f"节点 {idx}: 相关性分数 = {score:.4f}")
        
        logger.info(f"✅ 重排序完成，返回 {len(new_nodes)} 个节点")
        return new_nodes

