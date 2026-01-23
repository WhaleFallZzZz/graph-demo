"""
并行图谱后处理器
并行执行语义补偿和图谱上下文查询，减少数据库查询延迟。

主要功能：
1. 并行执行 SemanticEnrichment 的邻居查询
2. 并行执行 GraphContext 的路径计算
3. 合并结果并返回
4. 复用 Neo4j Session，减少连接开销
"""

import logging
from typing import List, Optional, Any
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class ParallelGraphPostprocessor(BaseNodePostprocessor):
    """
    并行图谱后处理器
    
    并行执行语义补偿和图谱上下文查询，减少延迟。
    使用 ThreadPoolExecutor 并行执行同步的数据库查询。
    """
    
    semantic_enricher: Any
    graph_context: Any
    max_workers: int = 2
    executor: Optional[Any] = None
    
    def __init__(
        self,
        semantic_enricher: Any,
        graph_context: Any,
        max_workers: int = 2,
        **kwargs: Any
    ) -> None:
        """
        初始化并行图谱后处理器
        
        Args:
            semantic_enricher: 语义补偿后处理器实例
            graph_context: 图谱上下文后处理器实例
            max_workers: 线程池最大工作线程数，默认为 2
            **kwargs: 其他传递给父类的参数
        """
        super().__init__(
            semantic_enricher=semantic_enricher,
            graph_context=graph_context,
            max_workers=max_workers,
            **kwargs
        )
        from concurrent.futures import ThreadPoolExecutor
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    @classmethod
    def class_name(cls) -> str:
        return "ParallelGraphPostprocessor"
    
    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """
        并行后处理节点
        
        Args:
            nodes: 节点列表
            query_bundle: 查询包
            
        Returns:
            List[NodeWithScore]: 处理后的节点列表
        """
        if not nodes:
            return nodes
        
        try:
            # 并行执行两个后处理器
            # 使用 ThreadPoolExecutor 来并行执行同步操作
            future_semantic = self.executor.submit(
                self._run_semantic_enrichment,
                nodes,
                query_bundle
            )
            future_graph = self.executor.submit(
                self._run_graph_context,
                nodes,
                query_bundle
            )
            
            # 等待两个任务完成
            semantic_result = future_semantic.result()
            graph_result = future_graph.result()
            
            # 合并结果
            # 语义补偿：添加邻居节点到原始节点列表
            # 图谱上下文：在列表前端添加路径节点
            result = semantic_result
            
            # 添加图谱上下文节点（如果有）
            if graph_result and len(graph_result) > 0:
                # 图谱上下文节点应该在列表前端
                # 检查 semantic_result 中是否已经包含了图谱上下文节点
                semantic_node_ids = {id(node.node) if hasattr(node, 'node') else id(node) for node in semantic_result}
                
                # 添加图谱上下文节点（去重）
                for graph_node in graph_result:
                    graph_node_id = id(graph_node.node) if hasattr(graph_node, 'node') else id(graph_node)
                    if graph_node_id not in semantic_node_ids:
                        result.insert(0, graph_node)  # 插入到前端
                        semantic_node_ids.add(graph_node_id)
            
            logger.info(f"并行后处理完成: 语义补偿={len(semantic_result)}, 图谱上下文={len(graph_result)}, 最终结果={len(result)}")
            return result
            
        except Exception as e:
            logger.error(f"并行后处理失败: {e}")
            import traceback
            logger.debug(f"错误堆栈: {traceback.format_exc()}")
            # 降级：串行执行
            return self._fallback_sequential(nodes, query_bundle)
    
    def _run_semantic_enrichment(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None
    ) -> List[NodeWithScore]:
        """执行语义补偿"""
        try:
            if self.semantic_enricher:
                return self.semantic_enricher._postprocess_nodes(nodes, query_bundle)
        except Exception as e:
            logger.warning(f"语义补偿执行失败: {e}")
        return nodes
    
    def _run_graph_context(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None
    ) -> List[NodeWithScore]:
        """执行图谱上下文"""
        try:
            if self.graph_context:
                result = self.graph_context._postprocess_nodes(nodes, query_bundle)
                
                # 提取图谱上下文节点（通常在列表前端）
                graph_context_nodes = []
                original_node_ids = {id(node.node) if hasattr(node, 'node') else id(node) for node in nodes}
                
                for node in result:
                    node_obj = node.node if hasattr(node, 'node') else node
                    node_id = id(node_obj)
                    
                    # 检查是否是新增的图谱上下文节点
                    if node_id not in original_node_ids:
                        metadata = getattr(node_obj, 'metadata', {}) or {}
                        if metadata.get('node_type') == 'graph_context':
                            graph_context_nodes.append(node)
                    else:
                        # 检查原始节点是否被标记为图谱上下文（不太可能，但处理边界情况）
                        metadata = getattr(node_obj, 'metadata', {}) or {}
                        if metadata.get('node_type') == 'graph_context':
                            graph_context_nodes.append(node)
                
                return graph_context_nodes
        except Exception as e:
            logger.warning(f"图谱上下文执行失败: {e}")
            import traceback
            logger.debug(f"错误堆栈: {traceback.format_exc()}")
        return []
    
    def _fallback_sequential(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None
    ) -> List[NodeWithScore]:
        """降级：串行执行"""
        logger.info("降级到串行执行")
        result = nodes
        
        # 先执行语义补偿
        try:
            if self.semantic_enricher:
                result = self.semantic_enricher._postprocess_nodes(result, query_bundle)
        except Exception as e:
            logger.warning(f"语义补偿失败: {e}")
        
        # 再执行图谱上下文
        try:
            if self.graph_context:
                result = self.graph_context._postprocess_nodes(result, query_bundle)
        except Exception as e:
            logger.warning(f"图谱上下文失败: {e}")
        
        return result
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
