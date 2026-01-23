#!/usr/bin/env python3
"""
硬匹配节点后处理器 - Hard Match Postprocessor

将查询前置处理中提取的硬匹配实体节点合并到向量检索结果中。
"""

import logging
from typing import List, Optional, Any
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle

logger = logging.getLogger(__name__)


class HardMatchPostprocessor(BaseNodePostprocessor):
    """
    硬匹配节点后处理器
    
    将硬匹配的实体节点添加到检索结果的顶部，确保重要实体不会被遗漏。
    
    Attributes:
        hard_match_nodes: 硬匹配的节点列表
    """
    
    hard_match_nodes: List[NodeWithScore]
    
    def __init__(
        self,
        hard_match_nodes: List[NodeWithScore],
        **kwargs: Any
    ):
        """
        初始化硬匹配后处理器
        
        Args:
            hard_match_nodes: 硬匹配的节点列表
            **kwargs: 其他传递给父类的参数
        """
        super().__init__(
            hard_match_nodes=hard_match_nodes or [],
            **kwargs
        )
    
    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """
        后处理节点：将硬匹配节点添加到结果顶部
        
        Args:
            nodes: 向量检索得到的节点列表
            query_bundle: 查询包（可选）
            
        Returns:
            List[NodeWithScore]: 合并后的节点列表（硬匹配节点在前）
        """
        if not self.hard_match_nodes:
            return nodes
        
        # 去重：检查硬匹配节点是否已在检索结果中
        existing_texts = set()
        for node_with_score in nodes:
            node = node_with_score.node
            text = node.get_content(metadata_mode="none")
            existing_texts.add(text)
        
        # 过滤掉已在检索结果中的硬匹配节点
        unique_hard_match = []
        for hard_match_node in self.hard_match_nodes:
            text = hard_match_node.node.get_content(metadata_mode="none")
            if text not in existing_texts:
                unique_hard_match.append(hard_match_node)
                existing_texts.add(text)
        
        if unique_hard_match:
            logger.info(f"添加 {len(unique_hard_match)} 个硬匹配节点到检索结果顶部")
            # 将硬匹配节点放在最前面
            return unique_hard_match + nodes
        
        return nodes
