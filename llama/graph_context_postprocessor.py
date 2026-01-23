"""
图谱上下文后处理器
在向量检索命中Top-K节点后，提取这些节点中的实体，
并在图谱中寻找连接这些实体的最短路径（子图），将路径信息转化为自然语言注入到上下文中。

支持两种优化策略：
1. 基于元路径 (Meta-path) 的搜索：根据查询意图搜索特定模式的路径（如：(疾病) -> [治疗] -> (药物)）
2. 社区发现：如果检索到的节点形成了密集社区，直接返回整个社区的核心结构
"""
import logging
from typing import List, Optional, Any, Dict, Set, Tuple, ClassVar
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode, MetadataMode
import itertools
import json

# 尝试导入 QueryIntent
try:
    from llama.query_intent import QueryIntent
except ImportError:
    try:
        from query_intent import QueryIntent
    except ImportError:
        from enum import Enum
        class QueryIntent(Enum):
            TREATMENT = "治疗防控"
            MECHANISM = "发病机制"
            SYMPTOM = "症状表现"
            DIAGNOSIS = "诊断检查"
            PREVENTION = "预防保健"
            COMPLICATION = "并发症"
            RISK_FACTOR = "风险因素"
            GENERAL = "综合查询"

logger = logging.getLogger(__name__)


class GraphContextPostprocessor(BaseNodePostprocessor):
    """
    图谱上下文后处理器（优化版）
    1. 提取检索节点中的实体
    2. 在图谱中寻找这些实体之间的连接路径（支持元路径和社区发现）
    3. 将子图结构转化为文本，作为上下文补充
    
    优化策略：
    - 基于元路径 (Meta-path) 的搜索：根据查询意图搜索特定模式的路径
    - 社区发现：如果节点形成密集社区，返回整个社区的核心结构
    """
    graph_store: Any
    max_path_depth: int = 2
    max_paths: int = 10
    query_intent: Optional[str] = None
    enable_community_detection: bool = True
    community_threshold: float = 0.3  # 社区密度阈值
    
    # 意图到元路径的映射（基于 GraphAgent 的 PATH_TYPES）
    # 使用 ClassVar 标记为类变量，避免被 Pydantic 视为模型字段
    INTENT_META_PATHS: ClassVar[Dict[QueryIntent, List[str]]] = {
        QueryIntent.TREATMENT: ["治疗", "防控", "矫正", "改善", "缓解"],
        QueryIntent.MECHANISM: ["导致", "引起", "形成", "发生", "表现为"],
        QueryIntent.SYMPTOM: ["表现为", "症状", "体征"],
        QueryIntent.DIAGNOSIS: ["检查", "测量", "诊断"],
        QueryIntent.PREVENTION: ["预防", "保健"],
        QueryIntent.COMPLICATION: ["并发症", "导致", "引起", "副作用"],
        QueryIntent.RISK_FACTOR: ["风险因素", "诱因", "相关"],
    }
    
    def __init__(
        self,
        graph_store: Any,
        max_path_depth: int = 2,
        max_paths: int = 10,
        query_intent: Optional[str] = None,
        enable_community_detection: bool = True,
        community_threshold: float = 0.3,
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
            query_intent=query_intent,
            enable_community_detection=enable_community_detection,
            community_threshold=community_threshold,
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
            
            # 2. 尝试社区发现（如果节点形成密集社区）
            is_neo4j = "Neo4jPropertyGraphStore" in str(type(self.graph_store))
            community_structure = None
            
            if self.enable_community_detection and len(entities) >= 3:
                community_structure = self._detect_community(entities, is_neo4j)
                if community_structure:
                    logger.info(f"发现密集社区，包含 {len(community_structure['core_entities'])} 个核心实体")
                    # 使用社区结构生成路径
                    paths = self._community_to_paths(community_structure, is_neo4j)
                else:
                    # 3. 寻找实体间的连接路径（支持元路径）
                    paths = self._find_subgraph_paths(entities, is_neo4j)
            else:
                # 3. 寻找实体间的连接路径（支持元路径）
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
        """寻找实体间的子图路径（支持元路径）"""
        found_paths = []
        
        # 限制实体对数量，避免组合爆炸 (取Top 10实体进行两两组合)
        top_entities = entities[:10]
        pairs = list(itertools.combinations(top_entities, 2))
        
        logger.debug(f"正在检查 {len(pairs)} 对实体间的连接...")
        
        # 获取元路径配置（如果提供了查询意图）
        meta_path_relations = self._get_meta_path_relations()
        
        if is_neo4j:
            return self._find_neo4j_paths(pairs, meta_path_relations)
        else:
            return self._find_memory_paths(pairs, meta_path_relations)
    
    def _get_meta_path_relations(self) -> Optional[List[str]]:
        """根据查询意图获取元路径关系类型"""
        if not self.query_intent:
            return None
        
        # 规范化意图并获取关系类型
        try:
            # 尝试直接匹配枚举值
            if isinstance(self.query_intent, str):
                # 尝试作为枚举值匹配
                for intent_enum in QueryIntent:
                    if intent_enum.value == self.query_intent or intent_enum.name == self.query_intent:
                        return self.INTENT_META_PATHS.get(intent_enum)
                # 尝试直接匹配
                for intent_enum, relations in self.INTENT_META_PATHS.items():
                    if intent_enum.value == self.query_intent:
                        return relations
            else:
                # 如果是枚举对象
                return self.INTENT_META_PATHS.get(self.query_intent)
        except Exception as e:
            logger.debug(f"获取元路径关系失败: {e}")
        
        return None

    def _find_neo4j_paths(self, pairs: List[Tuple[str, str]], meta_path_relations: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Neo4j 路径查找（支持元路径）"""
        paths = []
        try:
            with self.graph_store._driver.session() as session:
                for start, end in pairs:
                    if len(paths) >= self.max_paths:
                        break
                    
                    # 如果有元路径配置，优先使用元路径搜索
                    if meta_path_relations:
                        path = self._find_meta_path_neo4j(session, start, end, meta_path_relations)
                        if path:
                            paths.append(path)
                            logger.debug(f"发现元路径: {start} -> {end}")
                            continue
                    
                    # 否则使用最短路径
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
    
    def _find_meta_path_neo4j(self, session, start: str, end: str, meta_path_relations: List[str]) -> Optional[Dict[str, Any]]:
        """使用元路径在 Neo4j 中查找路径"""
        try:
            # 使用参数化查询，查找符合元路径模式的路径
            # 例如：(疾病) -> [治疗] -> (药物)
            query = f"""
            MATCH (start:__Entity__ {{name: $start_name}}), (end:__Entity__ {{name: $end_name}})
            MATCH path = (start)-[*1..{self.max_path_depth}]-(end)
            WHERE ALL(rel in relationships(path) WHERE type(rel) IN $meta_relations)
            RETURN [node in nodes(path) | node.name] as entity_names,
                   [rel in relationships(path) | rel.label] as relation_labels,
                   [rel in relationships(path) | type(rel)] as relations
            ORDER BY length(path) ASC
            LIMIT 1
            """
            result = session.run(query, start_name=start, end_name=end, meta_relations=meta_path_relations)
            record = result.single()
            
            if record:
                entity_names = record.get("entity_names")
                relations = record.get("relation_labels") or record.get("relations")
                
                if entity_names and relations:
                    return {
                        "entities": entity_names,
                        "relations": relations,
                        "source": start,
                        "target": end,
                        "is_meta_path": True
                    }
        except Exception as e:
            logger.debug(f"元路径查询失败: {e}")
        
        return None

    def _find_memory_paths(self, pairs: List[Tuple[str, str]], meta_path_relations: Optional[List[str]] = None) -> List[Dict[str, Any]]:
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
                            # 如果有元路径配置，只考虑符合元路径的关系
                            if meta_path_relations:
                                clean_rel = relation.replace("(反向)", "")
                                if clean_rel not in meta_path_relations:
                                    continue
                            
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
    
    def _detect_community(self, entities: List[str], is_neo4j: bool) -> Optional[Dict[str, Any]]:
        """
        检测实体是否形成密集社区
        
        使用简单的基于连接密度的社区检测算法：
        - 计算实体之间的连接密度
        - 如果密度超过阈值，认为形成社区
        - 返回社区的核心实体和连接结构
        
        Args:
            entities: 实体列表
            is_neo4j: 是否使用 Neo4j
            
        Returns:
            Optional[Dict[str, Any]]: 社区结构，包含：
                - core_entities: 核心实体列表
                - connections: 实体间的连接列表
                - density: 社区密度
        """
        if len(entities) < 3:
            return None
        
        try:
            if is_neo4j:
                return self._detect_community_neo4j(entities)
            else:
                return self._detect_community_memory(entities)
        except Exception as e:
            logger.debug(f"社区检测失败: {e}")
            return None
    
    def _detect_community_neo4j(self, entities: List[str]) -> Optional[Dict[str, Any]]:
        """在 Neo4j 中检测社区"""
        try:
            with self.graph_store._driver.session() as session:
                # 查询所有实体对之间的连接
                query = """
                MATCH (e1:__Entity__), (e2:__Entity__)
                WHERE e1.name IN $entities AND e2.name IN $entities
                  AND e1 <> e2
                  AND ((e1)-[*1..2]-(e2))
                RETURN DISTINCT e1.name as entity1, e2.name as entity2
                LIMIT 100
                """
                result = session.run(query, entities=entities)
                
                # 构建连接图
                connections = []
                entity_connections = {e: set() for e in entities}
                
                for record in result:
                    e1 = record["entity1"]
                    e2 = record["entity2"]
                    if e1 and e2:
                        connections.append((e1, e2))
                        entity_connections[e1].add(e2)
                        entity_connections[e2].add(e1)
                
                # 计算连接密度
                total_possible = len(entities) * (len(entities) - 1) / 2
                actual_connections = len(connections)
                density = actual_connections / total_possible if total_possible > 0 else 0
                
                logger.debug(f"社区检测: {len(entities)} 个实体, {actual_connections} 个连接, 密度: {density:.2f}")
                
                # 如果密度超过阈值，认为是社区
                if density >= self.community_threshold:
                    # 找出连接度最高的核心实体
                    core_entities = sorted(
                        entities,
                        key=lambda e: len(entity_connections[e]),
                        reverse=True
                    )[:min(5, len(entities))]  # 最多5个核心实体
                    
                    return {
                        "core_entities": core_entities,
                        "connections": connections,
                        "density": density,
                        "all_entities": entities
                    }
        except Exception as e:
            logger.debug(f"Neo4j 社区检测失败: {e}")
        
        return None
    
    def _detect_community_memory(self, entities: List[str]) -> Optional[Dict[str, Any]]:
        """在内存图中检测社区"""
        try:
            triplets = self.graph_store.get_triplets()
            entity_set = set(entities)
            
            # 构建连接图
            connections = []
            entity_connections = {e: set() for e in entities}
            
            for head, rel, tail in triplets:
                h_name = head.name if hasattr(head, 'name') else str(head)
                t_name = tail.name if hasattr(tail, 'name') else str(tail)
                
                if h_name in entity_set and t_name in entity_set and h_name != t_name:
                    connections.append((h_name, t_name))
                    entity_connections[h_name].add(t_name)
                    entity_connections[t_name].add(h_name)
            
            # 计算连接密度
            total_possible = len(entities) * (len(entities) - 1) / 2
            actual_connections = len(connections)
            density = actual_connections / total_possible if total_possible > 0 else 0
            
            logger.debug(f"社区检测: {len(entities)} 个实体, {actual_connections} 个连接, 密度: {density:.2f}")
            
            # 如果密度超过阈值，认为是社区
            if density >= self.community_threshold:
                # 找出连接度最高的核心实体
                core_entities = sorted(
                    entities,
                    key=lambda e: len(entity_connections[e]),
                    reverse=True
                )[:min(5, len(entities))]  # 最多5个核心实体
                
                return {
                    "core_entities": core_entities,
                    "connections": connections,
                    "density": density,
                    "all_entities": entities
                }
        except Exception as e:
            logger.debug(f"内存图社区检测失败: {e}")
        
        return None
    
    def _community_to_paths(self, community: Dict[str, Any], is_neo4j: bool) -> List[Dict[str, Any]]:
        """将社区结构转化为路径列表"""
        paths = []
        core_entities = community["core_entities"]
        connections = community["connections"]
        
        # 为核心实体之间的连接创建路径
        for i, e1 in enumerate(core_entities):
            for e2 in core_entities[i+1:]:
                # 检查是否有直接连接
                if (e1, e2) in connections or (e2, e1) in connections:
                    paths.append({
                        "entities": [e1, e2],
                        "relations": ["关联"],
                        "source": e1,
                        "target": e2,
                        "is_community": True
                    })
                    
                    if len(paths) >= self.max_paths:
                        break
            if len(paths) >= self.max_paths:
                break
        
        return paths
