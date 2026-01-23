#!/usr/bin/env python3
"""
限流重排序后处理器 - Limited Rerank Postprocessor

在重排序后只保留 Top N 个节点，实现漏斗式过滤。
"""

import logging
from typing import List, Optional, Any
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle

logger = logging.getLogger(__name__)


class LimitedRerankPostprocessor(BaseNodePostprocessor):
    """
    限流重排序后处理器
    
    先使用重排序器对节点进行重排序，然后只保留 Top N 个节点。
    用于实现漏斗式过滤策略：从大量候选节点中筛选出最相关的节点。
    
    Attributes:
        reranker: 底层重排序器实例
        top_n: 保留的节点数量
    """
    
    reranker: BaseNodePostprocessor
    top_n: int = 10
    
    def __init__(
        self,
        reranker: BaseNodePostprocessor,
        top_n: int = 10,
        **kwargs: Any
    ):
        """
        初始化限流重排序后处理器
        
        Args:
            reranker: 底层重排序器实例
            top_n: 保留的节点数量，默认为 10
            **kwargs: 其他传递给父类的参数
        """
        super().__init__(
            reranker=reranker,
            top_n=top_n,
            **kwargs
        )
    
    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """
        后处理节点：先重排序，然后只保留 Top N
        
        Args:
            nodes: 原始节点列表
            query_bundle: 查询包（可选）
            
        Returns:
            List[NodeWithScore]: 重排序后的 Top N 节点列表
        """
        if not nodes:
            return nodes
        
        # 1. 使用底层重排序器进行重排序
        try:
            reranked_nodes = self.reranker._postprocess_nodes(nodes, query_bundle)
            
            # 2. 只保留 Top N 个节点
            limited_nodes = reranked_nodes[:self.top_n]
            
            logger.info(f"重排序完成: {len(nodes)} 个节点 -> {len(reranked_nodes)} 个重排序节点 -> {len(limited_nodes)} 个 Top {self.top_n} 节点")
            
            return limited_nodes
            
        except Exception as e:
            logger.warning(f"限流重排序失败: {e}，返回原始节点")
            # 降级：直接截取前 Top N 个节点
            return nodes[:self.top_n]
