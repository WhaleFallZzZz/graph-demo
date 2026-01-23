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
from typing import List, Optional, Any, Dict, ClassVar
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode, MetadataMode
from llama_index.core.bridge.pydantic import PrivateAttr

# Python 3.9+ 兼容性：确保 Dict 和 List 是类型而不是函数
try:
    from typing import get_args, get_origin
    # 检查 Dict 是否是类型
    if not hasattr(Dict, '__origin__'):
        # 如果 Dict 是函数，使用 dict 替代
        Dict = dict
except ImportError:
    pass

try:
    from llama.query_intent import QueryIntent
except ImportError:
    try:
        from query_intent import QueryIntent
    except ImportError:
        # 降级处理：定义简单的枚举类
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
    query_intent: Optional[str] = None
    
    # 医学关键词列表（来自 enhanced_entity_extractor.StandardTermMapper）
    # 使用 ClassVar 标记为类变量，避免被 Pydantic 视为模型字段
    MEDICAL_KEYWORDS: ClassVar[List[str]] = [
        "近视", "远视", "散光", "弱视", "斜视", "病理性近视", "轴性近视", "屈光不正", "屈光参差",
        "眼轴长度", "屈光度", "调节幅度", "调节灵敏度", "眼压", "角膜曲率", "远视储备",
        "角膜塑形镜", "OK镜", "低浓度阿托品", "RGP镜片", "后巩膜加固术",
        "准分子激光手术", "LASIK", "全飞秒激光手术", "SMILE", "眼内接触镜植入", "ICL",
        "视网膜", "角膜", "晶状体", "视神经", "黄斑区", "脉络膜", "巩膜",
        "视物模糊", "视力下降", "豹纹状眼底", "视网膜萎缩", "脉络膜萎缩"
    ]
    
    # 私有属性，不作为 Pydantic 模型字段
    # 使用下划线前缀避免与字段名冲突
    # 注意：在 Python 3.9+ 中，如果 Dict 是函数，使用 dict 替代
    _intent_relation_filter: PrivateAttr = PrivateAttr(default_factory=dict)
    __base_intent_relation_filter: PrivateAttr = PrivateAttr(default_factory=dict)
    
    def __init__(
        self,
        graph_store: Any,
        max_neighbors_per_entity: int = 10,
        query_intent: Optional[str] = None,
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
            query_intent=query_intent,
            **kwargs
        )
        
        # 意图到关系类型的映射（用于过滤邻居关系）
        # 使用枚举值作为基础映射，然后建立多格式兼容映射
        base_relation_filter = {
            QueryIntent.TREATMENT: ["治疗", "防控", "矫正", "改善", "缓解", "治愈", "控制"],
            QueryIntent.MECHANISM: ["导致", "引起", "形成", "发生", "产生", "诱发"],
            QueryIntent.SYMPTOM: ["表现为", "症状", "体征", "显示", "出现"],
            QueryIntent.DIAGNOSIS: ["检查", "测量", "诊断", "检测", "筛查", "评估"],
            QueryIntent.PREVENTION: ["预防", "保健", "保护", "避免", "降低"],
            QueryIntent.COMPLICATION: ["并发症", "副作用", "导致", "引起", "风险"],
            QueryIntent.RISK_FACTOR: ["风险因素", "诱因", "相关", "影响", "关联"],
        }
        
        # 建立多格式兼容映射（支持枚举名、枚举值、中文值）
        self._intent_relation_filter = {}
        for intent_enum, relations in base_relation_filter.items():
            try:
                # 检查 intent_enum 是否是枚举对象
                if not hasattr(intent_enum, 'value') or not hasattr(intent_enum, 'name'):
                    logger.warning(f"意图枚举对象格式错误: {type(intent_enum)}, 跳过")
                    continue
                
                # 使用枚举值（中文）作为主键
                self._intent_relation_filter[intent_enum.value] = relations
                # 兼容枚举名（英文）
                self._intent_relation_filter[intent_enum.name] = relations
                # 兼容枚举对象本身（如果传入的是枚举）
                self._intent_relation_filter[intent_enum] = relations
            except Exception as e:
                logger.error(f"处理意图枚举 {intent_enum} 时出错: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue
        
        # 存储基础映射用于后续查询
        self.__base_intent_relation_filter = base_relation_filter
    
    @classmethod
    def class_name(cls) -> str:
        return "SemanticEnrichmentPostprocessor"
    
    def _normalize_and_get_relation_filter(self, query_intent: Optional[str]) -> Optional[List[str]]:
        """
        规范化查询意图并获取对应的关系过滤器
        
        支持多种意图格式：
        - 枚举对象：QueryIntent.TREATMENT
        - 枚举名（字符串）："TREATMENT"
        - 枚举值（字符串）："治疗防控"
        - 中文值：如果传入的是其他中文描述，尝试匹配
        
        Args:
            query_intent: 查询意图，可以是字符串或枚举对象
            
        Returns:
            Optional[List[str]]: 关系类型列表，如果找不到匹配的意图则返回 None
        """
        if not query_intent:
            return None
        
        # 情况1: 直接匹配（枚举对象、枚举名、枚举值）
        if query_intent in self._intent_relation_filter:
            return self._intent_relation_filter[query_intent]
        
        # 情况2: 尝试匹配枚举值（中文）
        if isinstance(query_intent, str):
            # 直接尝试字符串匹配
            if query_intent in self._intent_relation_filter:
                return self._intent_relation_filter[query_intent]
            
            # 情况3: 尝试通过枚举查找
            try:
                # 尝试作为枚举名查找
                if hasattr(QueryIntent, query_intent):
                    intent_enum = getattr(QueryIntent, query_intent)
                    if intent_enum in self._intent_relation_filter:
                        return self._intent_relation_filter[intent_enum]
                
                # 尝试作为枚举值查找（遍历所有枚举）
                for intent_enum in QueryIntent:
                    if intent_enum.value == query_intent:
                        if intent_enum in self._intent_relation_filter:
                            return self._intent_relation_filter[intent_enum]
                
            except Exception as e:
                logger.debug(f"意图规范化失败: {e}")
        
        # 未找到匹配的意图
        logger.debug(f"未找到意图 '{query_intent}' 的关系过滤器，使用所有关系")
        return None
    
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
                    # 查找该实体的一度关联节点（根据意图过滤关系）
                    neighbor_nodes = self._get_one_hop_neighbors(
                        entity, 
                        is_neo4j,
                        self.max_neighbors_per_entity,
                        query_intent=self.query_intent
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
        max_neighbors: int = 10,
        query_intent: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        获取实体的一度关联节点
        
        该方法查询知识图谱，找到与给定实体直接相连的所有节点（一度邻居）。
        支持根据查询意图过滤关系类型，只返回与意图相关的关系。
        支持两种图存储后端：Neo4j 和内存图存储。
        
        Args:
            entity_name: 实体名称，用于查询其关联节点
            is_neo4j: 是否使用 Neo4j 图存储
                      - True: 使用 Cypher 查询语言查询 Neo4j 数据库
                      - False: 遍历内存图存储中的三元组
            max_neighbors: 最多返回的关联节点数，默认为 10
                          用于控制查询结果的数量，避免返回过多数据
            query_intent: 查询意图（可选），用于过滤关系类型
                         如果提供，只返回与意图相关的关系
        
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
            - 如果提供了 query_intent，会根据意图过滤关系类型
            - 内存图查询遍历所有三元组，查找包含该实体的关系
            - 查询失败时会记录警告日志，但不会抛出异常
            - 返回的关联节点数量可能少于 max_neighbors，取决于实际关联节点数
        """
        neighbors = []
        
        # 根据查询意图获取需要的关系类型（支持多种格式）
        relation_filter = self._normalize_and_get_relation_filter(query_intent)
        if relation_filter:
            logger.debug(f"根据意图 '{query_intent}' 过滤关系类型: {relation_filter}")
        
        try:
            if is_neo4j:
                # Neo4j 参数化查询（消除注入风险）
                # 使用参数化查询确保安全性
                if relation_filter:
                    # 有关系过滤：使用参数化查询
                    query = """
                    MATCH (source:__Entity__ {name: $entity_name})-[r]->(target:__Entity__)
                    WHERE type(r) IN $relation_types
                    RETURN target.name as name, 
                           target.label as label,
                           type(r) as relation,
                           r.label as relation_label
                    LIMIT $limit
                    
                    UNION
                    
                    MATCH (source:__Entity__ {name: $entity_name})<-[r]-(target:__Entity__)
                    WHERE type(r) IN $relation_types
                    RETURN target.name as name,
                           target.label as label,
                           type(r) as relation,
                           r.label as relation_label
                    LIMIT $limit
                    """
                    
                    # 使用参数化查询，传递关系类型列表作为参数
                    with self.graph_store._driver.session() as session:
                        result = session.run(
                            query, 
                            entity_name=entity_name,
                            relation_types=relation_filter,  # 直接传递列表作为参数
                            limit=max_neighbors
                        )
                        # 在 session 关闭前收集所有结果
                        records = list(result)
                else:
                    # 不过滤，查询所有关系（参数化）
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
                    
                    # 使用参数化查询
                    with self.graph_store._driver.session() as session:
                        result = session.run(
                            query,
                            entity_name=entity_name,
                            limit=max_neighbors
                        )
                        # 在 session 关闭前收集所有结果
                        records = list(result)
                
                # 处理查询结果（在 session 关闭后）
                for record in records:
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
                    relation_type = type(relation).__name__ if hasattr(type(relation), '__name__') else str(relation)
                    
                    # 如果有关系过滤，检查关系类型是否匹配
                    if relation_filter:
                        if relation_label not in relation_filter and relation_type not in relation_filter:
                            continue
                    
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
