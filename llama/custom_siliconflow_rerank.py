"""
自定义 SiliconFlow Rerank 重排序器
"""
from typing import List, Optional, Any
import requests
import logging
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle

logger = logging.getLogger(__name__)

class CustomSiliconFlowRerank(BaseNodePostprocessor):
    """
    SiliconFlow 重排序器
    使用 SiliconFlow 的 Rerank API 进行重排序
    """
    
    api_key: str = Field(description="SiliconFlow API Key")
    model: str = Field(description="Rerank 模型名称")
    top_n: int = Field(default=2, description="返回的节点数量")
    base_url: str = Field(default="https://api.siliconflow.cn/v1/rerank", description="API 基础 URL")
    
    def __init__(
        self,
        api_key: str,
        model: str = "BAAI/bge-reranker-v2-m3",
        top_n: int = 2,
        base_url: str = "https://api.siliconflow.cn/v1/rerank",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
            top_n=top_n,
            base_url=base_url,
            **kwargs
        )

    @classmethod
    def class_name(cls) -> str:
        return "CustomSiliconFlowRerank"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        
        if not nodes:
            return []
            
        if query_bundle is None:
            raise ValueError("Query bundle must be provided.")
            
        query_str = query_bundle.query_str
        
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
        
        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            # 解析结果
            # 预期格式: {"results": [{"index": 0, "relevance_score": 0.9}, ...]}
            rerank_results = result.get("results", [])
            
            new_nodes = []
            for item in rerank_results:
                idx = item.get("index")
                score = item.get("relevance_score")
                
                if idx is not None and 0 <= idx < len(nodes):
                    node = nodes[idx]
                    node.score = score
                    new_nodes.append(node)
            
            return new_nodes
            
        except Exception as e:
            logger.error(f"SiliconFlow Rerank failed: {e}")
            # 如果失败，降级为返回原始节点的前top_n
            logger.warning("Falling back to original nodes sorting")
            return nodes[:self.top_n]
