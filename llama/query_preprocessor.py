#!/usr/bin/env python3
"""
查询前置处理模块 - Pre-Retrieval Query Processing

在向量检索之前对查询进行预处理，包括：
1. 查询意图分析
2. 根据意图改写查询（追加关键词）
3. 实体提取和硬匹配（Hard Match）
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from llama_index.core.schema import NodeWithScore, TextNode, MetadataMode

try:
    from .graph_agent import GraphAgent
    from .query_intent import QueryIntent
except ImportError:
    from graph_agent import GraphAgent
    from query_intent import QueryIntent

logger = logging.getLogger(__name__)


class QueryPreprocessor:
    """
    查询前置处理器
    
    在向量检索之前对查询进行预处理，提升检索质量和准确性。
    
    主要功能：
    1. 意图分析：使用 GraphAgent 分析查询意图
    2. 查询改写：根据意图类型自动追加相关关键词
    3. 实体提取：从查询中提取实体名称
    4. 硬匹配：检查实体是否在图谱中存在，返回对应节点
    
    Attributes:
        graph_agent: GraphAgent 实例，用于意图分析
        graph_store: 图谱存储对象
        is_neo4j: 是否为 Neo4j 图谱
    """
    
    # 意图相关的关键词增强映射
    INTENT_KEYWORD_ENHANCEMENT = {
        QueryIntent.TREATMENT: ["治疗方案", "临床效果", "疗效", "使用方法", "适应症"],
        QueryIntent.MECHANISM: ["发病机制", "原理", "病理生理", "形成过程"],
        QueryIntent.SYMPTOM: ["症状", "表现", "体征", "临床表现"],
        QueryIntent.DIAGNOSIS: ["诊断方法", "检查", "检测", "参数", "指标"],
        QueryIntent.PREVENTION: ["预防措施", "预防方法", "保健"],
        QueryIntent.COMPLICATION: ["并发症", "副作用", "风险", "不良反应"],
        QueryIntent.RISK_FACTOR: ["危险因素", "诱因", "相关因素"],
    }
    
    # 医学关键词列表（用于实体提取）
    MEDICAL_KEYWORDS = [
        # 疾病
        "近视", "远视", "散光", "弱视", "斜视", "病理性近视", "轴性近视", 
        "屈光不正", "屈光参差", "白内障", "青光眼", "沙眼",
        # 治疗手段
        "OK镜", "角膜塑形镜", "低浓度阿托品", "阿托品", "RGP镜片",
        "后巩膜加固术", "准分子激光手术", "LASIK", "全飞秒激光手术",
        "SMILE", "眼内接触镜植入", "ICL",
        # 解剖结构
        "视网膜", "角膜", "晶状体", "视神经", "黄斑区", "脉络膜", "巩膜",
        "眼轴", "前房", "房水", "悬韧带",
        # 症状
        "视物模糊", "视力下降", "豹纹状眼底", "视网膜萎缩", "脉络膜萎缩",
        # 检查参数
        "眼轴长度", "屈光度", "调节幅度", "调节灵敏度", "眼压", 
        "角膜曲率", "远视储备",
        # 其他
        "并发症", "调节", "集合", "融像"
    ]
    
    def __init__(self, graph_agent: GraphAgent, graph_store: Any):
        """
        初始化查询前置处理器
        
        Args:
            graph_agent: GraphAgent 实例，用于意图分析
            graph_store: 图谱存储对象
        """
        self.graph_agent = graph_agent
        self.graph_store = graph_store
        self.is_neo4j = "Neo4jPropertyGraphStore" in str(type(graph_store))
    
    def preprocess(self, query: str) -> Dict[str, Any]:
        """
        执行查询前置处理
        
        处理流程：
        1. 分析查询意图
        2. 根据意图改写查询（追加关键词）
        3. 从查询中提取实体
        4. 检查实体是否在图谱中存在
        5. 返回处理结果
        
        Args:
            query: 原始查询文本
            
        Returns:
            Dict[str, Any]: 处理结果，包含：
                - original_query: 原始查询
                - enhanced_query: 增强后的查询
                - intent: 查询意图
                - extracted_entities: 提取的实体列表
                - hard_match_nodes: 硬匹配的节点列表（NodeWithScore）
                - hard_match_entities: 在图谱中存在的实体列表
        """
        logger.info(f"开始查询前置处理: '{query}'")
        
        # 1. 分析查询意图
        intent = self.graph_agent.analyze_query_intent(query)
        logger.info(f"查询意图: {intent.value}")
        
        # 2. 根据意图改写查询
        enhanced_query = self._enhance_query(query, intent)
        if enhanced_query != query:
            logger.info(f"查询改写: '{query}' -> '{enhanced_query}'")
        
        # 3. 提取实体
        extracted_entities = self._extract_entities(query)
        logger.info(f"提取到实体: {extracted_entities}")
        
        # 4. 硬匹配：检查实体是否在图谱中存在
        hard_match_nodes, hard_match_entities = self._hard_match_entities(extracted_entities)
        if hard_match_entities:
            logger.info(f"硬匹配实体: {hard_match_entities} (共 {len(hard_match_nodes)} 个节点)")
        
        return {
            "original_query": query,
            "enhanced_query": enhanced_query,
            "intent": intent.value,
            "extracted_entities": extracted_entities,
            "hard_match_nodes": hard_match_nodes,
            "hard_match_entities": hard_match_entities
        }
    
    def _enhance_query(self, query: str, intent: QueryIntent) -> str:
        """
        根据意图改写查询，追加相关关键词
        
        Args:
            query: 原始查询
            intent: 查询意图
            
        Returns:
            str: 增强后的查询
        """
        # 如果是综合查询，不进行改写
        if intent == QueryIntent.GENERAL:
            return query
        
        # 获取该意图对应的关键词
        keywords = self.INTENT_KEYWORD_ENHANCEMENT.get(intent, [])
        
        if not keywords:
            return query
        
        # 检查查询中是否已包含这些关键词
        # 如果没有，追加最相关的关键词
        enhanced_parts = [query]
        
        # 检查是否包含比较类词汇（如"比较"、"区别"、"差异"）
        comparison_keywords = ["比较", "区别", "差异", "对比", "差别", "不同"]
        has_comparison = any(kw in query for kw in comparison_keywords)
        
        # 如果查询中包含比较类词汇，不追加意图关键词，避免干扰
        if not has_comparison:
            # 选择最相关的关键词追加（最多追加1-2个）
            for keyword in keywords[:2]:
                if keyword not in query:
                    enhanced_parts.append(keyword)
                    break
        
        return " ".join(enhanced_parts)
    
    def _extract_entities(self, query: str) -> List[str]:
        """
        从查询中提取实体
        
        使用 Neo4j 动态匹配方式从查询中提取图谱实体。
        优先使用 Neo4j 的全文索引或正则匹配，如果没有 Neo4j，则降级到关键词匹配。
        
        Args:
            query: 查询文本
            
        Returns:
            List[str]: 提取的实体列表
        """
        if self.is_neo4j:
            return self._extract_entities_from_neo4j(query)
        else:
            # 降级到关键词匹配（内存图谱）
            return self._extract_entities_by_keywords(query)
    
    def _extract_entities_from_neo4j(self, query: str) -> List[str]:
        """
        使用 Neo4j 动态匹配提取实体
        
        利用 Neo4j 的查询能力，直接在图谱中查找名称出现在查询中的实体。
        使用正则表达式或 CONTAINS 查询进行匹配。
        
        Args:
            query: 查询文本
            
        Returns:
            List[str]: 提取的实体列表
        """
        entities = []
        
        try:
            # 使用 Neo4j 的 CONTAINS 查询动态匹配实体
            # 直接查询：查找所有实体名称出现在查询中的实体
            # 注意：为了性能，限制查询长度
            if len(query) > 500:
                query = query[:500]
                logger.debug(f"查询文本过长，截断到500字符")
            
            # 转义查询中的单引号
            query_escaped = query.replace("'", "\\'")
            
            # 使用 CONTAINS 查询：查找实体名称出现在查询中的实体
            # Neo4j 的 CONTAINS 语法：string1 CONTAINS string2
            # 这里我们检查：query CONTAINS entity_name（查询中包含实体名）
            cypher_query = f"""
            MATCH (n:__Entity__)
            WHERE n.name IS NOT NULL
              AND size(n.name) >= 2
              AND '{query_escaped}' CONTAINS n.name
            RETURN DISTINCT n.name as name
            ORDER BY size(n.name) DESC
            LIMIT 10
            """
            
            try:
                result = self.graph_store.structured_query(cypher_query)
                matched_entities = set()
                
                for record in result:
                    entity_name = record.get('name', '').strip()
                    if entity_name and len(entity_name) >= 2:  # 过滤太短的实体
                        matched_entities.add(entity_name)
                
                entities = sorted(list(matched_entities), key=len, reverse=True)
                
                logger.info(f"Neo4j CONTAINS 查询找到 {len(entities)} 个实体: {entities[:5]}...")
                
            except Exception as e:
                logger.warning(f"Neo4j CONTAINS 查询失败: {e}，尝试正则匹配")
                
                # 降级方案：对查询进行分词，然后使用正则表达式匹配
                # 1. 提取长度 >= 2 的连续中文字符或英文单词
                clean_query = re.sub(r'[，。！？、；：""''（）【】\\s]+', ' ', query)
                tokens = re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z]{2,}', clean_query)
                
                if not tokens:
                    logger.debug("未找到有效的实体候选词")
                    return entities
                
                # 限制 token 数量以提高性能
                if len(tokens) > 20:
                    tokens = tokens[:20]
                    logger.debug(f"Token 数量过多，仅使用前20个")
                
                matched_entities = set()
                
                # 对每个 token 使用正则表达式匹配
                for token in tokens:
                    if len(token) < 2:
                        continue
                    
                    # 转义特殊字符（用于正则表达式）
                    token_escaped = re.escape(token)
                    
                    # 使用正则表达式：实体名称包含 token
                    regex_query = f"""
                    MATCH (n:__Entity__)
                    WHERE n.name IS NOT NULL
                      AND size(n.name) >= 2
                      AND n.name =~ '.*{token_escaped}.*'
                    RETURN DISTINCT n.name as name
                    ORDER BY size(n.name) DESC
                    LIMIT 5
                    """
                    
                    try:
                        result = self.graph_store.structured_query(regex_query)
                        for record in result:
                            entity_name = record.get('name', '').strip()
                            if entity_name and len(entity_name) >= 2:
                                matched_entities.add(entity_name)
                    except Exception as regex_error:
                        logger.warning(f"正则匹配失败 (token: {token}): {regex_error}")
                        continue
                
                entities = sorted(list(matched_entities), key=len, reverse=True)
                
                if len(entities) > 10:
                    entities = entities[:10]
                    logger.debug(f"实体数量过多，仅返回前10个")
                
                logger.info(f"Neo4j 正则匹配找到 {len(entities)} 个实体: {entities[:5]}...")
            
        except Exception as e:
            logger.warning(f"Neo4j 实体提取失败: {e}，降级到关键词匹配")
            entities = self._extract_entities_by_keywords(query)
        
        return entities
    
    def _extract_entities_by_keywords(self, query: str) -> List[str]:
        """
        使用关键词匹配提取实体（降级方案）
        
        当 Neo4j 不可用或查询失败时使用。
        
        Args:
            query: 查询文本
            
        Returns:
            List[str]: 提取的实体列表
        """
        entities = []
        
        # 按长度降序排序，优先匹配长实体（避免"近视"匹配到"病理性近视"的部分）
        sorted_keywords = sorted(self.MEDICAL_KEYWORDS, key=len, reverse=True)
        
        matched_positions = []  # 记录已匹配的位置，避免重复匹配
        
        for keyword in sorted_keywords:
            # 查找所有匹配位置
            positions = [m.start() for m in re.finditer(re.escape(keyword), query)]
            
            # 检查是否与已匹配的位置重叠
            for pos in positions:
                overlap = False
                start_pos = pos
                end_pos = pos + len(keyword)
                
                for matched_start, matched_end in matched_positions:
                    if not (end_pos <= matched_start or start_pos >= matched_end):
                        overlap = True
                        break
                
                if not overlap:
                    entities.append(keyword)
                    matched_positions.append((start_pos, end_pos))
                    break  # 每个关键词只匹配一次
        
        return entities
    
    def _hard_match_entities(self, entities: List[str]) -> Tuple[List[NodeWithScore], List[str]]:
        """
        硬匹配实体：检查实体是否在图谱中存在，返回对应节点
        
        Args:
            entities: 实体名称列表
            
        Returns:
            Tuple[List[NodeWithScore], List[str]]: 
                - 硬匹配的节点列表（NodeWithScore）
                - 在图谱中存在的实体列表
        """
        if not entities:
            return [], []
        
        hard_match_nodes = []
        hard_match_entities = []
        
        try:
            if self.is_neo4j:
                # Neo4j 查询
                # 限制实体数量以避免查询过长
                if len(entities) > 50:
                    entities = entities[:50]
                    logger.warning(f"实体列表过长，仅查询前50个")
                
                # 转义单引号，避免注入风险
                escaped_entities = [name.replace("'", "\\'") for name in entities]
                entity_names_str = "', '".join(escaped_entities)
                
                query = f"""
                MATCH (n:__Entity__)
                WHERE n.name IN ['{entity_names_str}']
                RETURN elementId(n) as id, n.name as name, n.type as type,
                       n.description as description, labels(n) as labels
                """
                
                result = self.graph_store.structured_query(query)
                
                for record in result:
                    entity_name = record.get('name', '')
                    entity_type = record.get('type', '概念')
                    entity_desc = record.get('description', '')
                    
                    # 创建 TextNode
                    text_content = f"{entity_name}（{entity_type}）"
                    if entity_desc:
                        text_content += f": {entity_desc}"
                    
                    node = TextNode(
                        text=text_content,
                        metadata={
                            "entity_name": entity_name,
                            "entity_type": entity_type,
                            "node_type": "entity",
                            "hard_match": True  # 标记为硬匹配
                        }
                    )
                    
                    # 创建 NodeWithScore，给予较高分数（确保在结果前列）
                    node_with_score = NodeWithScore(
                        node=node,
                        score=1.0  # 硬匹配节点给予最高分数
                    )
                    
                    hard_match_nodes.append(node_with_score)
                    hard_match_entities.append(entity_name)
            
            else:
                # 内存图谱查询
                try:
                    triplets = self.graph_store.get_triplets()
                    
                    # 构建实体名称集合
                    entity_set = set(entities)
                    
                    # 查找匹配的实体节点
                    matched_entities = set()
                    for triplet in triplets:
                        source = triplet[0]
                        target = triplet[2]
                        
                        source_name = getattr(source, 'name', str(source))
                        target_name = getattr(target, 'name', str(target))
                        
                        if source_name in entity_set:
                            matched_entities.add(source_name)
                            entity_set.remove(source_name)  # 避免重复
                            
                            # 创建节点
                            source_type = getattr(source, 'type', '概念')
                            node = TextNode(
                                text=f"{source_name}（{source_type}）",
                                metadata={
                                    "entity_name": source_name,
                                    "entity_type": source_type,
                                    "node_type": "entity",
                                    "hard_match": True
                                }
                            )
                            hard_match_nodes.append(NodeWithScore(node=node, score=1.0))
                        
                        if target_name in entity_set:
                            matched_entities.add(target_name)
                            entity_set.remove(target_name)
                            
                            target_type = getattr(target, 'type', '概念')
                            node = TextNode(
                                text=f"{target_name}（{target_type}）",
                                metadata={
                                    "entity_name": target_name,
                                    "entity_type": target_type,
                                    "node_type": "entity",
                                    "hard_match": True
                                }
                            )
                            hard_match_nodes.append(NodeWithScore(node=node, score=1.0))
                        
                        if not entity_set:
                            break
                    
                    hard_match_entities = list(matched_entities)
                    
                except Exception as e:
                    logger.warning(f"内存图谱硬匹配失败: {e}")
        
        except Exception as e:
            logger.warning(f"实体硬匹配失败: {e}")
        
        return hard_match_nodes, hard_match_entities
