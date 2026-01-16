"""
语义补偿后处理器

当向量检索找到实体时，自动拉取该实体在知识图谱中的一度关联节点作为语义补偿。

主要功能：
1. 从检索到的节点中提取实体名称
2. 查询每个实体在知识图谱中的一度关联节点
3. 将关联节点转换为文档节点，作为额外的上下文信息
4. 支持去重和数量限制

使用场景：
- 向量检索结果可能遗漏相关实体
- 通过一度关联节点提供更丰富的上下文信息
- 提高检索召回率和答案质量

依赖：
- LlamaIndex 的 BaseNodePostprocessor 基类
- Neo4j 或内存图存储
- enhanced_entity_extractor 中的 StandardTermMapper（可选）
"""

import logging
from typing import List, Optional, Any, Dict
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode, MetadataMode

logger = logging.getLogger(__name__)


class SemanticEnrichmentPostprocessor(BaseNodePostprocessor):
    """
    语义补偿后处理器
    
    从检索到的实体中提取一度关联节点，作为额外的上下文信息。
    
    工作流程：
    1. 从检索到的节点中提取实体名称
    2. 对每个实体，查询其在知识图谱中的一度关联节点
    3. 将关联节点转换为文档节点，添加到检索结果中
    4. 支持去重和数量限制，避免重复节点
    
    属性：
        graph_store: 图存储实例（Neo4j 或内存图）
        max_neighbors_per_entity: 每个实体最多拉取的关联节点数
    
    使用示例：
        ```python
        from llama.semantic_enrichment_postprocessor import SemanticEnrichmentPostprocessor
        
        # 创建后处理器
        enricher = SemanticEnrichmentPostprocessor(
            graph_store=graph_store,
            max_neighbors_per_entity=10
        )
        
        # 后处理检索结果
        enriched_nodes = enricher.postprocess_nodes(nodes, query_bundle)
        ```
    """
    graph_store: Any
    max_neighbors_per_entity: int = 10
    
    # 医学关键词列表（来自 enhanced_entity_extractor.StandardTermMapper）
    # 这些是眼科领域的常见术语，用于从文本中提取实体
    MEDICAL_KEYWORDS: List[str] = [
        "近视", "远视", "散光", "弱视", "斜视", "病理性近视", "轴性近视", "屈光不正", "屈光参差",
        "眼轴长度", "屈光度", "调节幅度", "调节灵敏度", "眼压", "角膜曲率", "远视储备",
        "角膜塑形镜", "OK镜", "低浓度阿托品", "RGP镜片", "后巩膜加固术",
        "准分子激光手术", "LASIK", "全飞秒激光手术", "SMILE", "眼内接触镜植入", "ICL",
        "视网膜", "角膜", "晶状体", "视神经", "黄斑区", "脉络膜", "巩膜",
        "视物模糊", "视力下降", "豹纹状眼底", "视网膜萎缩", "脉络膜萎缩"
    ]
    
    def __init__(
        self,
        graph_store: Any,
        max_neighbors_per_entity: int = 10,
        **kwargs: Any
    ) -> None:
        """
        初始化语义补偿后处理器
        
        Args:
            graph_store: 图存储实例，支持 Neo4j 或内存图存储
                        - Neo4j: 使用 Cypher 查询一度关联节点
                        - 内存图: 直接遍历三元组查找关联节点
            max_neighbors_per_entity: 每个实体最多拉取的关联节点数，默认为 10
                                    设置较大的值可以获取更多上下文，但会增加查询时间
            **kwargs: 其他传递给父类的参数
        
        Raises:
            无特定异常，初始化失败会在运行时记录警告日志
        
        Note:
            - 该后处理器会保留原始检索节点，只添加新的关联节点
            - 关联节点会被赋予较低的分数（0.3），表示这是补充信息
            - 如果图存储不可用，后处理器会静默失败，返回原始节点
        """
        super().__init__(
            graph_store=graph_store,
            max_neighbors_per_entity=max_neighbors_per_entity,
            **kwargs
        )
    
    @classmethod
    def class_name(cls) -> str:
        return "SemanticEnrichmentPostprocessor"
    
    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """
        后处理节点，添加一度关联节点作为语义补偿
        
        这是后处理器的主要方法，执行以下步骤：
        1. 检查输入节点和图存储是否可用
        2. 从检索到的节点中提取实体名称
        3. 为每个实体查找一度关联节点
        4. 将关联节点转换为文档节点并添加到结果中
        5. 使用去重机制避免重复添加相同的关联节点
        
        Args:
            nodes: 检索到的节点列表，每个节点包含文本和元数据
            query_bundle: 查询束，包含查询字符串和元数据（当前未使用，保留用于未来扩展）
        
        Returns:
            增强后的节点列表，包含原始节点和新增的关联节点
            - 原始节点保持不变，分数和内容不变
            - 关联节点会被赋予较低的分数（0.3），表示这是补充信息
            - 如果处理失败，返回原始节点列表
        
        Raises:
            无特定异常，所有异常都被捕获并记录日志，确保后处理器不会中断检索流程
        
        Note:
            - 使用 `added_nodes_set` 进行去重，避免重复添加相同的关联节点
            - 关联节点的唯一键格式为：`{neighbor_name}_{relation}`
            - 支持两种图存储：Neo4j（使用 Cypher 查询）和内存图（遍历三元组）
            - 处理失败时会记录详细的错误日志，包括堆栈跟踪
        """
        if not nodes or not self.graph_store:
            return nodes
        
        try:
            # 从检索到的节点中提取实体
            entities = self._extract_entities_from_nodes(nodes)
            
            if not entities:
                logger.debug("未找到实体，跳过语义补偿")
                return nodes
            
            logger.info(f"从检索结果中提取到 {len(entities)} 个实体: {entities[:5]}...")
            
            # 为每个实体查找一度关联节点
            enriched_nodes = list(nodes)  # 保留原始节点
            added_nodes_set = set()  # 用于去重
            
            is_neo4j = "Neo4jPropertyGraphStore" in str(type(self.graph_store))
            
            for entity in entities:
                try:
                    # 查找该实体的一度关联节点
                    neighbor_nodes = self._get_one_hop_neighbors(
                        entity, 
                        is_neo4j,
                        self.max_neighbors_per_entity
                    )
                    
                    # 将关联节点转换为文档节点
                    for neighbor_info in neighbor_nodes:
                        # 创建唯一的键用于去重
                        node_key = f"{neighbor_info['name']}_{neighbor_info.get('relation', '')}"
                        
                        if node_key not in added_nodes_set:
                            enriched_node = self._create_enrichment_node(neighbor_info, entity)
                            if enriched_node:
                                enriched_nodes.append(enriched_node)
                                added_nodes_set.add(node_key)
                                logger.debug(f"添加关联节点: {entity} -> {neighbor_info['name']} ({neighbor_info.get('relation', '')})")
                    
                except Exception as e:
                    logger.warning(f"获取实体 {entity} 的关联节点失败: {e}")
                    continue
            
            logger.info(f"语义补偿完成: 原始节点 {len(nodes)} 个，增强后 {len(enriched_nodes)} 个（新增 {len(enriched_nodes) - len(nodes)} 个关联节点）")
            
            return enriched_nodes
            
        except Exception as e:
            logger.error(f"语义补偿后处理失败: {e}")
            import traceback
            logger.debug(f"错误堆栈: {traceback.format_exc()}")
            # 失败时返回原始节点
            return nodes
    
    def _extract_entities_from_nodes(self, nodes: List[NodeWithScore]) -> List[str]:
        """
        从节点中提取实体名
        
        该方法尝试从两个来源提取实体：
        1. 节点的元数据中的 `entity_name` 字段
        2. 节点文本中的医学关键词匹配
        
        Args:
            nodes: 节点列表，每个节点包含文本和元数据
        
        Returns:
            实体名列表，去重后的实体名称
        
        Note:
            - 优先从元数据中提取实体名称
            - 如果元数据中没有实体名称，则使用医学关键词匹配
            - 医学关键词列表来自类属性 `MEDICAL_KEYWORDS`
            - 如果 `StandardTermMapper` 不可用，使用类属性中的医学关键词列表
            - 医学关键词包括：近视、远视、散光、弱视、斜视、眼轴长度、屈光度等
        """
        entities = []
        
        # 尝试导入 StandardTermMapper，但不是必需的
        try:
            from enhanced_entity_extractor import StandardTermMapper
        except ImportError:
            logger.debug("StandardTermMapper 未找到，使用类属性中的医学关键词列表")
        
        # 从节点的元数据或文本中提取实体
        for node_with_score in nodes:
            node = node_with_score.node
            
            # 从元数据中提取实体
            metadata = node.metadata or {}
            if 'entity_name' in metadata:
                entity = metadata['entity_name']
                if entity and entity not in entities:
                    entities.append(entity)
            
            # 从节点文本中提取实体
            node_text = node.get_content(metadata_mode=MetadataMode.NONE) if hasattr(node, 'get_content') else str(node)
            
            # 使用医学关键词匹配
            for keyword in self.MEDICAL_KEYWORDS:
                if keyword in node_text and keyword not in entities:
                    entities.append(keyword)
        
        return entities
    
    def _get_one_hop_neighbors(
        self, 
        entity_name: str, 
        is_neo4j: bool,
        max_neighbors: int = 10
    ) -> List[Dict[str, Any]]:
        """
        获取实体的一度关联节点
        
        该方法查询知识图谱，找到与给定实体直接相连的所有节点（一度邻居）。
        支持两种图存储后端：Neo4j 和内存图存储。
        
        Args:
            entity_name: 实体名称，用于查询其关联节点
            is_neo4j: 是否使用 Neo4j 图存储
                      - True: 使用 Cypher 查询语言查询 Neo4j 数据库
                      - False: 遍历内存图存储中的三元组
            max_neighbors: 最多返回的关联节点数，默认为 10
                          用于控制查询结果的数量，避免返回过多数据
        
        Returns:
            关联节点信息列表，每个元素是一个字典，包含：
            - name: 关联节点的名称
            - label: 关联节点的标签（类型）
            - relation: 关系类型
            - source_entity: 源实体名称（即输入的 entity_name）
            
            如果查询失败，返回空列表
        
        Note:
            - Neo4j 查询使用 Cypher 语言，查询两个方向的关系：
              * 出边：source -> target
              * 入边：source <- target
            - 内存图查询遍历所有三元组，查找包含该实体的关系
            - 查询失败时会记录警告日志，但不会抛出异常
            - 返回的关联节点数量可能少于 max_neighbors，取决于实际关联节点数
        """
        neighbors = []
        
        try:
            if is_neo4j:
                # Neo4j 查询
                # 使用 Cypher 查询语言查询一度关联节点
                # 查询两个方向的关系：出边和入边
                query = """
                MATCH (source:__Entity__ {name: $entity_name})-[r]->(target:__Entity__)
                RETURN target.name as name, 
                       target.label as label,
                       type(r) as relation,
                       r.label as relation_label
                LIMIT $limit
                
                UNION
                
                MATCH (source:__Entity__ {name: $entity_name})<-[r]-(target:__Entity__)
                RETURN target.name as name,
                       target.label as label,
                       type(r) as relation,
                       r.label as relation_label
                LIMIT $limit
                """
                
                with self.graph_store._driver.session() as session:
                    result = session.run(query, entity_name=entity_name, limit=max_neighbors)
                    for record in result:
                        # 优先使用关系标签，如果没有则使用关系类型
                        relation = record.get("relation_label") or record.get("relation") or "关联"
                        neighbors.append({
                            "name": record["name"],
                            "label": record.get("label", ""),
                            "relation": relation,
                            "source_entity": entity_name
                        })
            else:
                # 内存图存储查询
                # 遍历所有三元组，查找包含该实体的关系
                triplets = self.graph_store.get_triplets()
                
                # 查找包含该实体的三元组
                for head, relation, tail in triplets:
                    head_name = head.name if hasattr(head, 'name') else str(head)
                    tail_name = tail.name if hasattr(tail, 'name') else str(tail)
                    relation_label = relation.label if hasattr(relation, 'label') else str(relation)
                    
                    # 检查实体是否是头节点或尾节点
                    if head_name == entity_name:
                        neighbors.append({
                            "name": tail_name,
                            "label": tail.label if hasattr(tail, 'label') else "",
                            "relation": relation_label,
                            "source_entity": entity_name
                        })
                    elif tail_name == entity_name:
                        neighbors.append({
                            "name": head_name,
                            "label": head.label if hasattr(head, 'label') else "",
                            "relation": relation_label,
                            "source_entity": entity_name
                        })
                    
                    # 达到最大数量时停止
                    if len(neighbors) >= max_neighbors:
                        break
        
        except Exception as e:
            logger.warning(f"查询实体 {entity_name} 的关联节点失败: {e}")
        
        return neighbors[:max_neighbors]
    
    def _create_enrichment_node(self, neighbor_info: Dict[str, Any], source_entity: str) -> Optional[NodeWithScore]:
        """
        创建语义补偿节点
        
        该方法将关联节点信息转换为 LlamaIndex 的文档节点格式，
        用于添加到检索结果中作为补充上下文信息。
        
        Args:
            neighbor_info: 关联节点信息字典，包含：
                          - name: 关联节点的名称（必需）
                          - label: 关联节点的标签/类型（可选）
                          - relation: 关系类型（可选）
            source_entity: 源实体名称，即触发此关联节点的实体
        
        Returns:
            NodeWithScore 对象，包含：
            - node: TextNode 对象，包含文本内容和元数据
            - score: 节点分数，固定为 0.3，表示这是补充信息
            
            如果创建失败，返回 None
        
        Note:
            - 节点文本格式：`{neighbor_name}（{neighbor_label}）与{source_entity}存在{relation}关系`
            - 如果 neighbor_label 为空，则不显示标签部分
            - 如果 relation 为空，默认使用 "关联"
            - 节点元数据包含：
              * entity_name: 关联节点名称
              * entity_label: 关联节点标签
              * relation: 关系类型
              * source_entity: 源实体名称
              * enrichment_type: "one_hop_neighbor"
              * node_type: "semantic_enrichment"
            - 使用较低的分数（0.3）表示这是补充信息，不影响原始检索结果的排序
        """
        try:
            neighbor_name = neighbor_info.get("name", "")
            neighbor_label = neighbor_info.get("label", "")
            relation = neighbor_info.get("relation", "关联")
            
            # 构建节点文本内容
            text_parts = [f"{neighbor_name}"]
            if neighbor_label:
                text_parts.append(f"（{neighbor_label}）")
            text_parts.append(f"与{source_entity}存在{relation}关系")
            
            text = " ".join(text_parts)
            
            # 创建节点
            node = TextNode(
                text=text,
                metadata={
                    "entity_name": neighbor_name,
                    "entity_label": neighbor_label,
                    "relation": relation,
                    "source_entity": source_entity,
                    "enrichment_type": "one_hop_neighbor",
                    "node_type": "semantic_enrichment"
                }
            )
            
            # 使用较低的分数，表示这是补充信息
            return NodeWithScore(node=node, score=0.3)
            
        except Exception as e:
            logger.warning(f"创建语义补偿节点失败: {e}")
            return None
