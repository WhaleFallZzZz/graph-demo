"""
图谱上下文后处理器
在向量检索命中Top-K节点后，提取这些节点中的实体，
并在图谱中寻找连接这些实体的最短路径（子图），将路径信息转化为自然语言注入到上下文中。
"""
import logging
from typing import List, Optional, Any, Dict, Set, Tuple
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode, MetadataMode
import itertools

logger = logging.getLogger(__name__)


class GraphContextPostprocessor(BaseNodePostprocessor):
    """
    图谱上下文后处理器
    1. 提取检索节点中的实体
    2. 在图谱中寻找这些实体之间的连接路径（Shortest Path）
    3. 将子图结构转化为文本，作为上下文补充
    """
    graph_store: Any
    max_path_depth: int = 2
    max_paths: int = 10
    
    def __init__(
        self,
        graph_store: Any,
        max_path_depth: int = 2,
        max_paths: int = 10,
        **kwargs: Any
    ) -> None:
        """
        初始化
        
        Args:
            graph_store: 图存储实例
            max_path_depth: 搜索路径的最大深度 (默认2，即 A->B->C)
            max_paths: 最多返回的路径数量
        """
        super().__init__(
            graph_store=graph_store,
            max_path_depth=max_path_depth,
            max_paths=max_paths,
            **kwargs
        )
    
    @classmethod
    def class_name(cls) -> str:
        return "GraphContextPostprocessor"
    
    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """
        后处理节点
        """
        if not nodes or not self.graph_store:
            return nodes
        
        try:
            # 1. 从Top-K节点中提取实体
            entities = self._extract_entities_from_nodes(nodes)
            
            if not entities or len(entities) < 2:
                logger.debug("实体数量不足以建立连接，跳过图谱上下文增强")
                return nodes
            
            logger.info(f"Top-K节点提取实体: {len(entities)}个 -> {entities[:5]}...")
            
            # 2. 寻找实体间的连接路径 (Subgraph Extraction)
            is_neo4j = "Neo4jPropertyGraphStore" in str(type(self.graph_store))
            paths = self._find_subgraph_paths(entities, is_neo4j)
            
            if not paths:
                logger.info("未发现Top-K实体间的图谱连接")
                return nodes
            
            logger.info(f"发现 {len(paths)} 条图谱连接路径，正在生成上下文...")
            
            # 3. 将路径转化为自然语言上下文节点
            graph_context_node = self._create_graph_context_node(paths)
            
            if graph_context_node:
                # 将图谱上下文节点添加到列表前端，使其获得更高的注意力权重
                return [graph_context_node] + nodes
            
            return nodes
            
        except Exception as e:
            logger.error(f"图谱上下文后处理失败: {e}")
            import traceback
            logger.debug(f"错误堆栈: {traceback.format_exc()}")
            return nodes
    
    def _extract_entities_from_nodes(self, nodes: List[NodeWithScore]) -> List[str]:
        """从节点中提取实体名"""
        entities = set()
        
        # 复用医学关键词列表
        medical_keywords = [
            "近视", "远视", "散光", "弱视", "斜视", "病理性近视", "轴性近视", "屈光不正", "屈光参差",
            "眼轴长度", "屈光度", "调节幅度", "调节灵敏度", "眼压", "角膜曲率", "远视储备",
            "角膜塑形镜", "OK镜", "低浓度阿托品", "RGP镜片", "后巩膜加固术",
            "准分子激光手术", "LASIK", "全飞秒激光手术", "SMILE", "眼内接触镜植入", "ICL",
            "视网膜", "角膜", "晶状体", "视神经", "黄斑区", "脉络膜", "巩膜",
            "视物模糊", "视力下降", "豹纹状眼底", "视网膜萎缩", "脉络膜萎缩", "并发症"
        ]
        
        for node_with_score in nodes:
            node = node_with_score.node
            
            # 1. 尝试从元数据获取
            metadata = node.metadata or {}
            if 'entity_name' in metadata:
                entities.add(metadata['entity_name'])
            
            # 2. 尝试从文本内容提取
            text = node.get_content(metadata_mode=MetadataMode.NONE)
            for kw in medical_keywords:
                if kw in text:
                    entities.add(kw)
                    
        return list(entities)

    def _find_subgraph_paths(self, entities: List[str], is_neo4j: bool) -> List[Dict[str, Any]]:
        """寻找实体间的子图路径"""
        found_paths = []
        
        # 限制实体对数量，避免组合爆炸 (取Top 10实体进行两两组合)
        top_entities = entities[:10]
        pairs = list(itertools.combinations(top_entities, 2))
        
        logger.debug(f"正在检查 {len(pairs)} 对实体间的连接...")
        
        if is_neo4j:
            return self._find_neo4j_paths(pairs)
        else:
            return self._find_memory_paths(pairs)

    def _find_neo4j_paths(self, pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """Neo4j 路径查找"""
        paths = []
        try:
            with self.graph_store._driver.session() as session:
                for start, end in pairs:
                    if len(paths) >= self.max_paths:
                        break
                        
                    # 查找最短路径
                    query = f"""
                    MATCH (start:__Entity__), (end:__Entity__)
                    WHERE start.name = $start_name AND end.name = $end_name
                    MATCH path = shortestPath((start)-[*1..{self.max_path_depth}]-(end))
                    RETURN path, 
                           [node in nodes(path) | node.name] as entity_names,
                           [rel in relationships(path) | type(rel)] as relations,
                           [rel in relationships(path) | rel.label] as relation_labels
                    LIMIT 1
                    """
                    result = session.run(query, start_name=start, end_name=end)
                    record = result.single()
                    
                    if record:
                        entity_names = record.get("entity_names")
                        relations = record.get("relation_labels") or record.get("relations")
                        
                        if entity_names and relations:
                            paths.append({
                                "entities": entity_names,
                                "relations": relations,
                                "source": start,
                                "target": end
                            })
                            logger.debug(f"发现路径: {start} -> {end}")
                            
        except Exception as e:
            logger.warning(f"Neo4j 子图搜索失败: {e}")
            
        return paths

    def _find_memory_paths(self, pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """内存图路径查找 (简单 BFS)"""
        paths = []
        try:
            triplets = self.graph_store.get_triplets()
            adjacency = {}
            
            # 构建邻接表
            for head, rel, tail in triplets:
                h_name = head.name if hasattr(head, 'name') else str(head)
                t_name = tail.name if hasattr(tail, 'name') else str(tail)
                r_label = rel.label if hasattr(rel, 'label') else str(rel)
                
                if h_name not in adjacency: adjacency[h_name] = []
                if t_name not in adjacency: adjacency[t_name] = []
                
                adjacency[h_name].append((t_name, r_label))
                # 视为无向图进行搜索，增加连通率
                adjacency[t_name].append((h_name, r_label + "(反向)"))

            for start, end in pairs:
                if len(paths) >= self.max_paths: break
                
                # BFS
                queue = [(start, [start], [])] # (current, nodes, relations)
                visited = {start}
                found = False
                
                while queue:
                    curr, node_path, rel_path = queue.pop(0)
                    if len(node_path) > self.max_path_depth + 1: continue
                    
                    if curr == end:
                        paths.append({
                            "entities": node_path,
                            "relations": rel_path,
                            "source": start,
                            "target": end
                        })
                        found = True
                        break
                    
                    if curr in adjacency:
                        for neighbor, relation in adjacency[curr]:
                            if neighbor not in visited:
                                visited.add(neighbor)
                                queue.append((neighbor, node_path + [neighbor], rel_path + [relation]))
                    
                    if found: break
                    
        except Exception as e:
            logger.warning(f"内存图子图搜索失败: {e}")
            
        return paths

    def _create_graph_context_node(self, paths: List[Dict[str, Any]]) -> Optional[NodeWithScore]:
        """将路径转化为自然语言节点"""
        if not paths:
            return None
            
        text_lines = ["【图谱逻辑结构】(基于知识图谱的推理依据):"]
        
        for p in paths:
            entities = p["entities"]
            relations = p["relations"]
            
            # 格式化路径
            # LLM 上下文格式: A --[关联]--> B
            context_str = entities[0]
            # 前端展示格式: A->关联->B (用户期望格式)
            display_str = entities[0]

            for i, rel in enumerate(relations):
                if i + 1 < len(entities):
                    rel_text = str(rel) if rel is not None else "关联"
                    clean_rel = rel_text.replace("(反向)", "")
                    
                    context_str += f" --[{clean_rel}]--> {entities[i+1]}"
                    display_str += f"->{clean_rel}->{entities[i+1]}"
            
            # 保存格式化后的路径字符串到 path 对象中
            p["path_str"] = display_str
            
            text_lines.append(f"- {context_str}")
            
        context_text = "\n".join(text_lines)
        
        # 创建节点
        # 设置 node_type="semantic_enrichment" 以便前端的 _extract_paths_from_source_nodes 能够识别并提取
        # 注意：之前的 _extract_paths_from_source_nodes 逻辑是基于 one-hop neighbor 的简单 metadata
        # 这里我们是 complex path。
        # 为了兼容前端复用，我们需要一种巧妙的方式让 _extract_paths_from_source_nodes 也能解析这个复杂路径
        # 或者我们只是为了 Prompt 注入，而前端展示仍通过 `graph_data` 事件传递。
        # 既然 _extract_paths_from_source_nodes 是我在上一步为了复用 source_nodes 而写的，
        # 我可以在 metadata 中存储完整路径信息，然后在那里进行解析。
        
        metadata = {
            "node_type": "graph_context",
            "path_count": len(paths),
            "paths_data": json.dumps(paths, ensure_ascii=False) # 序列化路径数据供提取
        }
        
        node = TextNode(text=context_text, metadata=metadata)
        
        # 给予最高置信度，确保 LLM 优先参考
        return NodeWithScore(node=node, score=1.0)

import json
