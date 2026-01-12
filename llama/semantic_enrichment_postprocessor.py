"""
语义补偿后处理器
当向量检索找到实体时，自动拉取该实体在图谱中的一度关联节点作为语义补偿
"""
import logging
from typing import List, Optional, Any
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode, MetadataMode

logger = logging.getLogger(__name__)


class SemanticEnrichmentPostprocessor(BaseNodePostprocessor):
    """
    语义补偿后处理器
    从检索到的实体中提取一度关联节点，作为额外的上下文信息
    """
    
    def __init__(
        self,
        graph_store: Any,
        max_neighbors_per_entity: int = 10,
        **kwargs: Any
    ) -> None:
        """
        初始化语义补偿后处理器
        
        Args:
            graph_store: 图存储实例
            max_neighbors_per_entity: 每个实体最多拉取的关联节点数
        """
        super().__init__(**kwargs)
        self.graph_store = graph_store
        self.max_neighbors_per_entity = max_neighbors_per_entity
    
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
        
        Args:
            nodes: 检索到的节点列表
            query_bundle: 查询束
            
        Returns:
            增强后的节点列表（包含原始节点和关联节点）
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
        从节点中提取实体名（使用 StandardTermMapper）
        
        Args:
            nodes: 节点列表
            
        Returns:
            实体名列表
        """
        entities = []
        
        try:
            from enhanced_entity_extractor import StandardTermMapper
        except ImportError:
            logger.warning("StandardTermMapper 未找到，使用简单提取")
            StandardTermMapper = None
        
        # 医学关键词列表（来自 StandardTermMapper）
        medical_keywords = [
            "近视", "远视", "散光", "弱视", "斜视", "病理性近视", "轴性近视", "屈光不正", "屈光参差",
            "眼轴长度", "屈光度", "调节幅度", "调节灵敏度", "眼压", "角膜曲率", "远视储备",
            "角膜塑形镜", "OK镜", "低浓度阿托品", "RGP镜片", "后巩膜加固术",
            "准分子激光手术", "LASIK", "全飞秒激光手术", "SMILE", "眼内接触镜植入", "ICL",
            "视网膜", "角膜", "晶状体", "视神经", "黄斑区", "脉络膜", "巩膜",
            "视物模糊", "视力下降", "豹纹状眼底", "视网膜萎缩", "脉络膜萎缩"
        ]
        
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
            for keyword in medical_keywords:
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
        
        Args:
            entity_name: 实体名称
            is_neo4j: 是否使用 Neo4j
            max_neighbors: 最多返回的关联节点数
            
        Returns:
            关联节点信息列表
        """
        neighbors = []
        
        try:
            if is_neo4j:
                # Neo4j 查询
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
                        relation = record.get("relation_label") or record.get("relation") or "关联"
                        neighbors.append({
                            "name": record["name"],
                            "label": record.get("label", ""),
                            "relation": relation,
                            "source_entity": entity_name
                        })
            else:
                # 内存图存储查询
                triplets = self.graph_store.get_triplets()
                
                # 查找包含该实体的三元组
                for head, relation, tail in triplets:
                    head_name = head.name if hasattr(head, 'name') else str(head)
                    tail_name = tail.name if hasattr(tail, 'name') else str(tail)
                    relation_label = relation.label if hasattr(relation, 'label') else str(relation)
                    
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
                    
                    if len(neighbors) >= max_neighbors:
                        break
        
        except Exception as e:
            logger.warning(f"查询实体 {entity_name} 的关联节点失败: {e}")
        
        return neighbors[:max_neighbors]
    
    def _create_enrichment_node(self, neighbor_info: Dict[str, Any], source_entity: str) -> Optional[NodeWithScore]:
        """
        创建语义补偿节点
        
        Args:
            neighbor_info: 关联节点信息
            source_entity: 源实体名
            
        Returns:
            节点对象
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
