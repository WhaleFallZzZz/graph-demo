#!/usr/bin/env python3
"""
智能图谱查询代理 - Graph-Agent
根据图谱中的连接密度自主决定查询路径
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import re

try:
    from .query_intent import QueryIntent
except ImportError:
    from query_intent import QueryIntent

logger = logging.getLogger(__name__)


class ConnectionDensity:
    """
    连接密度信息类
    
    用于存储和分析图谱中实体的连接密度信息，包括：
    - 总连接数
    - 各关系类型的连接数量
    - 各邻居类型的连接数量
    - 连接的邻居实体列表
    
    主要用途：
    - 分析实体的连接特征
    - 决定查询路径的优先级
    - 优化图谱查询策略
    
    Attributes:
        entity (str): 实体名称
        total_connections (int): 总连接数
        connection_types (Dict[str, int]): 关系类型及其连接数，格式为 {relation_type: count}
        neighbor_types (Dict[str, int]): 邻居类型及其数量，格式为 {entity_type: count}
        neighbor_entities (List[str]): 连接的邻居实体列表
    """
    
    def __init__(self, entity: str):
        """
        初始化连接密度对象
        
        Args:
            entity: 实体名称
        """
        self.entity = entity
        self.total_connections = 0
        self.connection_types = {}  # relation_type -> count
        self.neighbor_types = {}  # entity_type -> count
        self.neighbor_entities = []  # list of connected entities
        
    def add_connection(self, relation: str, neighbor_type: str, neighbor_entity: str):
        """
        添加连接信息
        
        Args:
            relation: 关系类型
            neighbor_type: 邻居实体类型
            neighbor_entity: 邻居实体名称
        """
        self.total_connections += 1
        self.connection_types[relation] = self.connection_types.get(relation, 0) + 1
        self.neighbor_types[neighbor_type] = self.neighbor_types.get(neighbor_type, 0) + 1
        if neighbor_entity not in self.neighbor_entities:
            self.neighbor_entities.append(neighbor_entity)
    
    def get_dominant_relation(self) -> Optional[str]:
        """
        获取主导关系类型
        
        Returns:
            连接数最多的关系类型，如果没有连接则返回 None
        """
        if not self.connection_types:
            return None
        return max(self.connection_types.items(), key=lambda x: x[1])[0]
    
    def get_dominant_neighbor_type(self) -> Optional[str]:
        """
        获取主导邻居类型
        
        Returns:
            连接数最多的邻居类型，如果没有连接则返回 None
        """
        if not self.neighbor_types:
            return None
        return max(self.neighbor_types.items(), key=lambda x: x[1])[0]


class GraphAgent:
    """
    智能图谱查询代理
    
    根据图谱中的连接密度和查询意图自主决定查询路径，实现智能化的图谱查询。
    
    主要功能：
    - 查询意图分析：使用关键词匹配和 LLM 分类器分析用户查询意图
    - 连接密度探测：分析实体在图谱中的连接特征
    - 智能路径决策：根据意图和连接密度选择最优查询路径
    - 多路径查询执行：支持 Neo4j 和内存图谱的查询执行
    - 结果合并与聚合：合并多个路径的查询结果
    
    支持的查询意图：
    - TREATMENT: 治疗、防控、矫正等
    - MECHANISM: 发病机制、原理等
    - SYMPTOM: 症状、表现、体征等
    - DIAGNOSIS: 诊断、检查、检测等
    - PREVENTION: 预防、保健等
    - COMPLICATION: 并发症、副作用等
    - RISK_FACTOR: 风险因素、诱因等
    - GENERAL: 综合查询
    
    Attributes:
        graph_store: 图谱存储对象（Neo4j 或内存图谱）
        is_neo4j (bool): 是否为 Neo4j 图谱
        cache (Dict[str, ConnectionDensity]): 连接密度缓存
        use_llm_classifier (bool): 是否使用 LLM 意图分类器
        llm: LLM 实例
        llm_classifier: LLM 意图分类器实例
    """
    
    # 查询意图关键词映射
    INTENT_KEYWORDS = {
        QueryIntent.TREATMENT: [
            "治疗", "防控", "控制", "矫正", "改善", "缓解", "方案", "方法", "措施",
            "阿托品", "OK镜", "塑形镜", "RGP", "眼镜", "手术", "激光", "训练"
        ],
        QueryIntent.MECHANISM: [
            "机制", "原理", "原因", "导致", "引起", "形成", "发生", "发展", "病理",
            "生理", "眼轴", "屈光", "调节", "集合", "融像"
        ],
        QueryIntent.SYMPTOM: [
            "症状", "表现", "体征", "症状体征", "视物模糊", "视力下降", "看不清",
            "疲劳", "不适", "感觉", "现象"
        ],
        QueryIntent.DIAGNOSIS: [
            "诊断", "检查", "检测", "测量", "参数", "指标", "筛查", "发现",
            "眼压", "角膜曲率", "屈光度", "眼轴长度", "视力"
        ],
        QueryIntent.PREVENTION: [
            "预防", "保健", "保护", "注意", "避免", "减少", "降低", "增强",
            "户外活动", "用眼卫生", "休息"
        ],
        QueryIntent.COMPLICATION: [
            "并发症", "副作用", "风险", "危害", "影响", "不良", "后遗症",
            "视网膜脱落", "青光眼", "白内障"
        ],
        QueryIntent.RISK_FACTOR: [
            "风险因素", "诱因", "危险", "因素", "影响", "相关", "关联",
            "遗传", "环境", "习惯"
        ]
    }
    
    # 路径类型定义
    PATH_TYPES = {
        QueryIntent.TREATMENT: [
            {"relation": "治疗", "direction": "outgoing"},
            {"relation": "防控", "direction": "outgoing"},
            {"relation": "矫正", "direction": "outgoing"},
            {"relation": "改善", "direction": "outgoing"},
            {"relation": "缓解", "direction": "outgoing"}
        ],
        QueryIntent.MECHANISM: [
            {"relation": "导致", "direction": "outgoing"},
            {"relation": "引起", "direction": "outgoing"},
            {"relation": "形成", "direction": "outgoing"},
            {"relation": "发生", "direction": "outgoing"},
            {"relation": "表现为", "direction": "outgoing"}
        ],
        QueryIntent.SYMPTOM: [
            {"relation": "表现为", "direction": "outgoing"},
            {"relation": "症状", "direction": "outgoing"},
            {"relation": "体征", "direction": "outgoing"}
        ],
        QueryIntent.DIAGNOSIS: [
            {"relation": "检查", "direction": "outgoing"},
            {"relation": "测量", "direction": "outgoing"},
            {"relation": "诊断", "direction": "outgoing"}
        ],
        QueryIntent.PREVENTION: [
            {"relation": "预防", "direction": "outgoing"},
            {"relation": "保健", "direction": "outgoing"}
        ],
        QueryIntent.COMPLICATION: [
            {"relation": "并发症", "direction": "outgoing"},
            {"relation": "导致", "direction": "outgoing"},
            {"relation": "引起", "direction": "outgoing"},
            {"relation": "副作用", "direction": "outgoing"}
        ],
        QueryIntent.RISK_FACTOR: [
            {"relation": "风险因素", "direction": "outgoing"},
            {"relation": "诱因", "direction": "outgoing"},
            {"relation": "相关", "direction": "outgoing"}
        ]
    }
    
    # 意图权重调整规则
    # 当查询中包含特定关键词时，提升对应意图的权重
    INTENT_WEIGHT_ADJUSTMENTS = {
        QueryIntent.TREATMENT: {
            "keywords": ["阿托品", "OK镜", "塑形镜", "RGP", "眼镜", "手术", "激光"],
            "weight_boost": 2
        },
        QueryIntent.DIAGNOSIS: {
            "keywords": ["眼压", "角膜曲率", "屈光度", "眼轴长度", "视力"],
            "weight_boost": 2
        },
        QueryIntent.MECHANISM: {
            "keywords": ["眼轴", "屈光", "调节", "集合", "融像"],
            "weight_boost": 1
        },
        QueryIntent.COMPLICATION: {
            "keywords": ["视网膜脱落", "青光眼", "白内障"],
            "weight_boost": 2
        }
    }
    
    # 意图优先级
    # 当多个意图得分相同时，按优先级选择（数字越小优先级越高）
    INTENT_PRIORITY = {
        QueryIntent.TREATMENT: 1,
        QueryIntent.DIAGNOSIS: 2,
        QueryIntent.MECHANISM: 3,
        QueryIntent.SYMPTOM: 4,
        QueryIntent.PREVENTION: 5,
        QueryIntent.COMPLICATION: 6,
        QueryIntent.RISK_FACTOR: 7,
        QueryIntent.GENERAL: 999
    }
    
    def __init__(self, graph_store, llm_instance=None, use_llm_classifier=True):
        """
        初始化 Graph-Agent
        
        Args:
            graph_store: 图谱存储对象，支持 Neo4j 和内存图谱
            llm_instance: LLM 实例，用于意图分类（可选）
            use_llm_classifier: 是否使用 LLM 意图分类器，默认为 True
                              如果为 False，则仅使用关键词匹配进行意图分析
        """
        self.graph_store = graph_store
        self.is_neo4j = "Neo4jPropertyGraphStore" in str(type(graph_store))
        self.cache = {}  # 缓存连接密度信息
        self._node_map_cache = None  # 缓存节点映射
        self._triplets_cache = None  # 缓存三元组
        self.use_llm_classifier = use_llm_classifier
        self.llm = llm_instance
        
        # 初始化LLM意图分类器
        if use_llm_classifier:
            try:
                try:
                    from .llm_intent_classifier import LLMIntentClassifier
                except ImportError:
                    from llm_intent_classifier import LLMIntentClassifier
                self.llm_classifier = LLMIntentClassifier(llm_instance)
                logger.info("✅ LLM意图分类器初始化成功")
            except ImportError as e:
                logger.warning(f"⚠️ 无法导入LLM意图分类器: {e}，将使用关键词匹配方式")
                self.llm_classifier = None
                self.use_llm_classifier = False
        else:
            self.llm_classifier = None
        
    def analyze_query_intent(self, query: str) -> QueryIntent:
        """
        分析查询意图
        
        使用 LLM 分类器或关键词匹配分析用户查询的意图。
        优先使用 LLM 分类器，如果失败则降级到关键词匹配。
        
        意图分析流程：
        1. 如果启用了 LLM 分类器，优先使用 LLM 进行意图分类
        2. 如果 LLM 分类失败，降级到关键词匹配
        3. 关键词匹配计算每个意图的匹配分数
        4. 应用权重调整规则（根据特定关键词提升对应意图的权重）
        5. 如果多个意图得分相同，按优先级选择
        
        Args:
            query: 用户查询文本
            
        Returns:
            QueryIntent: 查询意图枚举值，表示识别出的查询意图
        """
        # 如果启用了LLM分类器，优先使用
        if self.use_llm_classifier and self.llm_classifier:
            try:
                intent, confidence, reasoning = self.llm_classifier.classify(query)
                logger.info(f"LLM意图分类: '{query}' -> {intent.value} (置信度: {confidence:.2f})")
                logger.debug(f"推理过程: {reasoning}")
                return intent
            except Exception as e:
                logger.warning(f"LLM意图分类失败，降级到关键词匹配: {e}")
        
        # 降级到关键词匹配方式
        query_lower = query.lower()
        intent_scores = {}
        
        # 计算每个意图的匹配分数
        for intent, keywords in self.INTENT_KEYWORDS.items():
            score = 0
            for keyword in keywords:
                if keyword in query:
                    score += 1
            intent_scores[intent] = score
        
        # 应用权重调整规则（配置化）
        for intent, adjustment in self.INTENT_WEIGHT_ADJUSTMENTS.items():
            for keyword in adjustment["keywords"]:
                if keyword in query:
                    intent_scores[intent] += adjustment["weight_boost"]
                    logger.debug(f"权重调整: 意图 {intent.value} 因关键词 '{keyword}' 提升 {adjustment['weight_boost']} 分")
                    break  # 每个意图只应用一次权重提升
        
        # 找出得分最高的意图
        max_score = max(intent_scores.values())
        
        # 如果没有匹配到任何意图，返回综合查询
        if max_score == 0:
            return QueryIntent.GENERAL
        
        # 如果有多个意图得分相同，按优先级选择
        top_intents = [intent for intent, score in intent_scores.items() if score == max_score]
        
        if len(top_intents) > 1:
            # 按优先级排序，选择优先级最高的
            sorted_intents = sorted(top_intents, key=lambda x: self.INTENT_PRIORITY.get(x, 999))
            selected_intent = sorted_intents[0]
            logger.info(f"多个意图得分相同 ({max_score})，按优先级选择: {selected_intent.value}")
        else:
            selected_intent = top_intents[0]
        
        logger.info(f"关键词匹配意图分析: '{query}' -> {selected_intent.value} (得分: {max_score})")
        return selected_intent
    
    def detect_connection_density(self, entity: str, max_depth: int = 2) -> ConnectionDensity:
        """
        探测实体的连接密度
        
        分析实体在图谱中的连接特征，包括连接数、关系类型、邻居类型等。
        支持缓存以提高性能。
        
        探测流程：
        1. 检查缓存，如果已存在则直接返回
        2. 根据图谱类型（Neo4j 或内存图谱）选择探测方法
        3. 查询实体的直接连接（一阶邻居）
        4. 如果需要深度探测，查询二阶连接
        5. 统计连接密度信息并缓存结果
        
        Args:
            entity: 实体名称
            max_depth: 探测深度，默认为 2
                      1 表示只查询直接连接
                      2 表示查询直接连接和二阶连接
            
        Returns:
            ConnectionDensity: 连接密度对象，包含实体的连接统计信息
        """
        cache_key = f"{entity}_{max_depth}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        density = ConnectionDensity(entity)
        
        try:
            if self.is_neo4j:
                self._detect_neo4j_density(density, entity, max_depth)
            else:
                self._detect_memory_density(density, entity, max_depth)
        except Exception as e:
            logger.warning(f"探测连接密度失败: {e}")
        
        self.cache[cache_key] = density
        logger.debug(f"连接密度探测: {entity} -> {density.total_connections} 连接, "
                    f"主导关系: {density.get_dominant_relation()}, "
                    f"主导邻居: {density.get_dominant_neighbor_type()}")
        
        return density
    
    def _detect_neo4j_density(self, density: ConnectionDensity, entity: str, max_depth: int):
        """
        探测 Neo4j 图谱的连接密度
        
        使用 Cypher 查询语言查询 Neo4j 图谱中实体的连接信息。
        
        查询内容：
        1. 查询实体的直接连接（一阶邻居）
        2. 如果需要深度探测，查询二阶连接
        
        Args:
            density: ConnectionDensity 对象，用于存储连接密度信息
            entity: 实体名称
            max_depth: 探测深度
        """
        try:
            # 查询实体的直接连接
            query = f"""
            MATCH (n {{name: '{entity}'}})-[r]->(m)
            RETURN type(r) as relation, m.name as neighbor, labels(m)[0] as neighbor_type
            """
            result = self.graph_store.structured_query(query)
            
            for record in result:
                relation = record.get('relation', '')
                neighbor = record.get('neighbor', '')
                neighbor_type = record.get('neighbor_type', 'unknown')
                density.add_connection(relation, neighbor_type, neighbor)
            
            # 如果需要深度探测，查询二阶连接
            if max_depth >= 2:
                query2 = f"""
                MATCH (n {{name: '{entity}'}})-[r1]->(m)-[r2]->(p)
                RETURN type(r1) as relation1, type(r2) as relation2, 
                       m.name as neighbor, labels(m)[0] as neighbor_type,
                       p.name as second_neighbor, labels(p)[0] as second_neighbor_type
                LIMIT 50
                """
                result2 = self.graph_store.structured_query(query2)
                
                for record in result2:
                    relation1 = record.get('relation1', '')
                    neighbor = record.get('neighbor', '')
                    neighbor_type = record.get('neighbor_type', 'unknown')
                    density.add_connection(relation1, neighbor_type, neighbor)
        
        except Exception as e:
            logger.warning(f"Neo4j 连接密度探测失败: {e}")
    
    def _detect_memory_density(self, density: ConnectionDensity, entity: str, max_depth: int):
        """
        探测内存图谱的连接密度
        
        使用 SimplePropertyGraphStore 的 get_triplets() 方法查询内存图谱中实体的连接信息。
        
        查询流程：
        1. 获取所有三元组（source, relation, target）
        2. 查找与实体相关的连接（作为 source 或 target）
        3. 统计连接密度信息
        
        Args:
            density: ConnectionDensity 对象，用于存储连接密度信息
            entity: 实体名称
            max_depth: 探测深度（当前版本仅支持一阶连接）
        """
        try:
            # SimplePropertyGraphStore 使用 get_triplets() 获取所有三元组
            triplets = self.graph_store.get_triplets()
            
            # 查找与实体相关的连接
            for triplet in triplets:
                source = triplet[0]
                relation = triplet[1]
                target = triplet[2]
                
                # 获取节点名称和类型
                source_name = getattr(source, 'name', str(source))
                target_name = getattr(target, 'name', str(target))
                source_type = getattr(source, 'type', 'unknown')
                target_type = getattr(target, 'type', 'unknown')
                relation_label = getattr(relation, 'label', str(relation))
                
                if source_name == entity:
                    density.add_connection(relation_label, target_type, target_name)
                elif target_name == entity:
                    density.add_connection(relation_label, source_type, source_name)
        
        except Exception as e:
            logger.warning(f"内存图谱连接密度探测失败: {e}")
    
    def decide_query_paths(self, query: str, entities: List[str]) -> List[Dict[str, Any]]:
        """
        根据查询意图和连接密度决定查询路径
        
        智能决策查询路径，结合查询意图和实体连接密度信息。
        
        决策流程：
        1. 分析查询意图
        2. 探测每个实体的连接密度
        3. 根据意图和连接密度决定路径
           - 综合查询（GENERAL）：探索多个路径
           - 特定意图：优先选择相关路径
        4. 按优先级排序路径
        
        Args:
            query: 用户查询文本
            entities: 查询实体列表
            
        Returns:
            List[Dict[str, Any]]: 查询路径列表，每个路径包含：
                - type: 路径类型
                - description: 路径描述
                - priority: 优先级（0-1，越高越优先）
                - query_params: 查询参数
        """
        paths = []
        
        if not entities:
            logger.warning("没有提供查询实体，使用默认路径")
            return self._get_default_paths()
        
        # 1. 分析查询意图
        intent = self.analyze_query_intent(query)
        
        # 2. 探测每个实体的连接密度
        entity_densities = {}
        for entity in entities:
            density = self.detect_connection_density(entity, max_depth=2)
            entity_densities[entity] = density
        
        # 3. 根据意图和连接密度决定路径
        if intent == QueryIntent.GENERAL:
            # 综合查询：探索多个路径
            paths = self._decide_general_paths(entities, entity_densities)
        else:
            # 特定意图：优先选择相关路径
            paths = self._decise_intent_paths(intent, entities, entity_densities)
        
        # 4. 按优先级排序
        paths.sort(key=lambda x: x['priority'], reverse=True)
        
        logger.info(f"决定查询路径: {len(paths)} 条路径, 意图: {intent.value}")
        for i, path in enumerate(paths):
            logger.debug(f"  路径 {i+1}: {path['description']} (优先级: {path['priority']})")
        
        return paths
    
    def _decide_general_paths(self, entities: List[str], 
                              densities: Dict[str, ConnectionDensity]) -> List[Dict[str, Any]]:
        """
        决定综合查询的路径
        
        根据实体的主导关系类型决定查询路径。
        
        路径决策规则：
        - 主导关系为治疗/防控/矫正：选择治疗防控路径
        - 主导关系为导致/引起/形成：选择发病机制路径
        - 主导关系为表现为/症状/体征：选择症状表现路径
        - 其他：选择通用路径
        
        Args:
            entities: 实体列表
            densities: 实体到连接密度的映射
            
        Returns:
            List[Dict[str, Any]]: 查询路径列表
        """
        paths = []
        
        for entity, density in densities.items():
            # 根据连接密度决定路径
            dominant_relation = density.get_dominant_relation()
            
            if dominant_relation in ["治疗", "防控", "矫正"]:
                # 治疗防控路径
                paths.append({
                    "type": "treatment",
                    "description": f"{entity}的治疗防控路径",
                    "priority": 0.8,
                    "query_params": {
                        "entity": entity,
                        "relations": ["治疗", "防控", "矫正", "改善", "缓解"],
                        "direction": "outgoing",
                        "max_depth": 2
                    }
                })
            elif dominant_relation in ["导致", "引起", "形成"]:
                # 发病机制路径
                paths.append({
                    "type": "mechanism",
                    "description": f"{entity}的发病机制路径",
                    "priority": 0.8,
                    "query_params": {
                        "entity": entity,
                        "relations": ["导致", "引起", "形成", "发生"],
                        "direction": "outgoing",
                        "max_depth": 2
                    }
                })
            elif dominant_relation in ["表现为", "症状", "体征"]:
                # 症状表现路径
                paths.append({
                    "type": "symptom",
                    "description": f"{entity}的症状表现路径",
                    "priority": 0.7,
                    "query_params": {
                        "entity": entity,
                        "relations": ["表现为", "症状", "体征"],
                        "direction": "outgoing",
                        "max_depth": 2
                    }
                })
            else:
                # 默认路径
                paths.append({
                    "type": "general",
                    "description": f"{entity}的通用路径",
                    "priority": 0.5,
                    "query_params": {
                        "entity": entity,
                        "relations": [],
                        "direction": "both",
                        "max_depth": 2
                    }
                })
        
        return paths
    
    def _decise_intent_paths(self, intent: QueryIntent, entities: List[str],
                            densities: Dict[str, ConnectionDensity]) -> List[Dict[str, Any]]:
        """
        根据特定意图决定路径
        
        根据查询意图和实体的连接特征决定查询路径。
        
        路径决策规则：
        - 如果实体有相关连接，优先级更高（0.9）
        - 如果实体没有相关连接，优先级较低（0.6）
        
        Args:
            intent: 查询意图
            entities: 实体列表
            densities: 实体到连接密度的映射
            
        Returns:
            List[Dict[str, Any]]: 查询路径列表
        """
        paths = []
        
        # 获取意图对应的路径类型
        path_configs = self.PATH_TYPES.get(intent, [])
        
        for entity, density in densities.items():
            # 检查实体是否有相关连接
            entity_connections = density.connection_types
            
            for path_config in path_configs:
                relation = path_config['relation']
                direction = path_config['direction']
                
                # 如果实体有相关连接，优先级更高
                if relation in entity_connections:
                    priority = 0.9
                    description = f"{entity}的{intent.value}路径 (有{relation}连接)"
                else:
                    priority = 0.6
                    description = f"{entity}的{intent.value}路径 (探索{relation})"
                
                paths.append({
                    "type": intent.value,
                    "description": description,
                    "priority": priority,
                    "query_params": {
                        "entity": entity,
                        "relations": [relation],
                        "direction": direction,
                        "max_depth": 2
                    }
                })
        
        return paths
    
    def _get_default_paths(self) -> List[Dict[str, Any]]:
        """
        获取默认路径
        
        当没有提供查询实体时，返回默认的查询路径配置。
        
        Returns:
            List[Dict[str, Any]]: 默认查询路径列表，包含一个通用路径配置
        """
        return [{
            "type": "default",
            "description": "默认查询路径",
            "priority": 0.5,
            "query_params": {
                "entity": "",
                "relations": [],
                "direction": "both",
                "max_depth": 2
            }
        }]
    
    def execute_graph_query(self, path: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        执行图谱查询
        
        根据查询路径配置执行图谱查询，支持 Neo4j 和内存图谱。
        
        执行流程：
        1. 根据图谱类型选择查询方法
        2. 执行查询并获取结果
        3. 记录查询日志
        
        Args:
            path: 查询路径配置，包含：
                - description: 路径描述
                - query_params: 查询参数
                    - entity: 查询实体
                    - relations: 关系类型列表
                    - direction: 查询方向（outgoing/incoming/both）
                    - max_depth: 查询深度
            
        Returns:
            List[Dict[str, Any]]: 查询结果列表，每个结果包含：
                - source: 源实体
                - relation: 关系
                - target: 目标实体
                - target_type: 目标实体类型
        """
        results = []
        params = path['query_params']
        
        try:
            if self.is_neo4j:
                results = self._execute_neo4j_query(params)
            else:
                results = self._execute_memory_query(params)
            
            logger.debug(f"执行查询: {path['description']}, 返回 {len(results)} 条结果")
        except Exception as e:
            logger.warning(f"执行查询失败: {e}")
        
        return results
    
    def _execute_neo4j_query(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        执行 Neo4j 查询
        
        使用 Cypher 查询语言查询 Neo4j 图谱。
        
        查询类型：
        - 特定关系查询：根据关系类型和方向查询
        - 通用查询：查询所有连接
        
        查询参数：
        - entity: 查询实体名称
        - relations: 关系类型列表（可选）
        - direction: 查询方向（outgoing/incoming/both）
        - max_depth: 查询深度（1 或 2）
        
        Args:
            params: 查询参数字典
            
        Returns:
            List[Dict[str, Any]]: 查询结果列表
        """
        results = []
        entity = params.get('entity', '')
        relations = params.get('relations', [])
        direction = params.get('direction', 'both')
        max_depth = params.get('max_depth', 2)
        
        if not entity:
            return results
        
        try:
            if relations:
                # 特定关系查询
                relation_filter = "|".join(relations)
                if direction == "outgoing":
                    if max_depth == 1:
                        query = f"""
                        MATCH (n {{name: '{entity}'}})-[r:{relation_filter}]->(m)
                        RETURN n.name as source, type(r) as relation, m.name as target, 
                               labels(m)[0] as target_type
                        LIMIT 50
                        """
                    else:
                        query = f"""
                        MATCH (n {{name: '{entity}'}})-[r:{relation_filter}*1..{max_depth}]->(m)
                        RETURN n.name as source, [rel in r | type(rel)] as relations, m.name as target, 
                               labels(m)[0] as target_type
                        LIMIT 50
                        """
                elif direction == "incoming":
                    if max_depth == 1:
                        query = f"""
                        MATCH (n)-[r:{relation_filter}]->(m {{name: '{entity}'}})
                        RETURN n.name as source, type(r) as relation, m.name as target,
                               labels(n)[0] as source_type
                        LIMIT 50
                        """
                    else:
                        query = f"""
                        MATCH (n)-[r:{relation_filter}*1..{max_depth}]->(m {{name: '{entity}'}})
                        RETURN n.name as source, [rel in r | type(rel)] as relations, m.name as target,
                               labels(n)[0] as source_type
                        LIMIT 50
                        """
                else:
                    if max_depth == 1:
                        query = f"""
                        MATCH (n {{name: '{entity}'}})-[r:{relation_filter}]-(m)
                        RETURN n.name as source, type(r) as relation, m.name as target,
                               labels(m)[0] as target_type
                        LIMIT 50
                        """
                    else:
                        query = f"""
                        MATCH (n {{name: '{entity}'}})-[r:{relation_filter}*1..{max_depth}]-(m)
                        RETURN n.name as source, [rel in r | type(rel)] as relations, m.name as target,
                               labels(m)[0] as target_type
                        LIMIT 50
                        """
            else:
                # 通用查询
                if max_depth == 1:
                    query = f"""
                    MATCH (n {{name: '{entity}'}})-[r]-(m)
                    RETURN n.name as source, type(r) as relation, m.name as target,
                           labels(m)[0] as target_type
                    LIMIT 50
                    """
                else:
                    query = f"""
                    MATCH (n {{name: '{entity}'}})-[r*1..{max_depth}]-(m)
                    RETURN n.name as source, [rel in r | type(rel)] as relations, m.name as target,
                           labels(m)[0] as target_type
                    LIMIT 50
                    """
            
            result = self.graph_store.structured_query(query)
            
            for record in result:
                # 处理单步和多步查询的不同结果格式
                relation = record.get('relation', '')
                relations_list = record.get('relations', [])
                
                # 如果是多步查询，将关系列表转换为字符串
                if relations_list:
                    relation = ' -> '.join(relations_list) if isinstance(relations_list, list) else str(relations_list)
                
                results.append({
                    "source": record.get('source', ''),
                    "relation": relation,
                    "target": record.get('target', ''),
                    "target_type": record.get('target_type', 'unknown')
                })
        
        except Exception as e:
            logger.warning(f"Neo4j 查询失败: {e}")
        
        return results
    
    def _execute_memory_query(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        执行内存图谱查询
        
        使用 SimplePropertyGraphStore 的 get_triplets() 方法查询内存图谱。
        
        查询流程：
        1. 获取所有三元组（使用缓存）
        2. 构建节点映射（使用缓存）
        3. 查找与实体相关的连接
        4. 根据关系类型过滤结果
        
        查询参数：
        - entity: 查询实体名称
        - relations: 关系类型列表（可选）
        - max_depth: 查询深度（当前版本仅支持一阶连接）
        
        Args:
            params: 查询参数字典
            
        Returns:
            List[Dict[str, Any]]: 查询结果列表
        """
        results = []
        entity = params.get('entity', '')
        relations = params.get('relations', [])
        max_depth = params.get('max_depth', 2)
        
        if not entity:
            return results
        
        try:
            # 使用缓存的三元组和节点映射
            if self._triplets_cache is None or self._node_map_cache is None:
                triplets = self.graph_store.get_triplets()
                self._triplets_cache = triplets
                
                # 构建节点映射（name -> node）
                node_map = {}
                for triplet in triplets:
                    source = triplet[0]
                    target = triplet[2]
                    
                    # 获取节点名称
                    source_name = getattr(source, 'name', str(source))
                    target_name = getattr(target, 'name', str(target))
                    
                    if source_name not in node_map:
                        node_map[source_name] = source
                    if target_name not in node_map:
                        node_map[target_name] = target
                
                self._node_map_cache = node_map
            else:
                triplets = self._triplets_cache
                node_map = self._node_map_cache
            
            # 查找相关关系
            for triplet in triplets:
                source = triplet[0]
                relation = triplet[1]
                target = triplet[2]
                
                # 获取节点名称和类型
                source_name = getattr(source, 'name', str(source))
                target_name = getattr(target, 'name', str(target))
                source_type = getattr(source, 'type', 'unknown')
                target_type = getattr(target, 'type', 'unknown')
                relation_label = getattr(relation, 'label', str(relation))
                
                # 检查关系过滤
                if relations and relation_label not in relations:
                    continue
                
                # 查找与实体相关的连接
                if source_name == entity:
                    results.append({
                        "source": source_name,
                        "relation": relation_label,
                        "target": target_name,
                        "target_type": target_type
                    })
                elif target_name == entity:
                    results.append({
                        "source": source_name,
                        "relation": relation_label,
                        "target": target_name,
                        "target_type": source_type
                    })
        
        except Exception as e:
            logger.warning(f"内存图谱查询失败: {e}")
            import traceback
            logger.debug(f"内存图谱查询错误堆栈: {traceback.format_exc()}")
        
        return results
    
    def merge_path_results(self, path_results: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        合并多个路径的查询结果
        
        将多个查询路径的结果合并为一个统一的结果结构，便于后续处理。
        
        合并内容：
        - 路径统计：总路径数、总结果数
        - 实体统计：各实体的出现次数
        - 关系统计：各关系的出现次数
        - 路径详情：每个路径的结果统计
        
        Args:
            path_results: 多个路径的查询结果列表，每个元素是一个路径的结果列表
            
        Returns:
            Dict[str, Any]: 合并后的结果，包含：
                - total_paths: 总路径数
                - total_results: 总结果数
                - entities: 实体及其出现次数的字典
                - relations: 关系及其出现次数的字典
                - paths: 每个路径的详细信息列表
        """
        merged = {
            "total_paths": len(path_results),
            "total_results": sum(len(r) for r in path_results),
            "entities": {},
            "relations": {},
            "paths": []
        }
        
        # 收集所有实体和关系
        for i, results in enumerate(path_results):
            path_info = {
                "path_index": i,
                "result_count": len(results),
                "entities": set(),
                "relations": set()
            }
            
            for result in results:
                source = result.get('source', '')
                target = result.get('target', '')
                relation = result.get('relation', '')
                
                if source:
                    merged["entities"][source] = merged["entities"].get(source, 0) + 1
                    path_info["entities"].add(source)
                if target:
                    merged["entities"][target] = merged["entities"].get(target, 0) + 1
                    path_info["entities"].add(target)
                if relation:
                    merged["relations"][relation] = merged["relations"].get(relation, 0) + 1
                    path_info["relations"].add(relation)
            
            path_info["entities"] = list(path_info["entities"])
            path_info["relations"] = list(path_info["relations"])
            merged["paths"].append(path_info)
        
        logger.info(f"合并路径结果: {merged['total_paths']} 条路径, "
                   f"{merged['total_results']} 条结果, "
                   f"{len(merged['entities'])} 个实体, "
                   f"{len(merged['relations'])} 种关系")
        
        return merged
    
    def query(self, query: str, entities: List[str]) -> Dict[str, Any]:
        """
        执行智能图谱查询
        
        这是 GraphAgent 的主要入口方法，整合了意图分析、路径决策、查询执行和结果合并。
        
        查询流程：
        1. 分析查询意图
        2. 决定查询路径（根据意图和连接密度）
        3. 执行查询（支持多路径并行查询）
        4. 合并结果
        
        Args:
            query: 用户查询文本
            entities: 查询实体列表
            
        Returns:
            Dict[str, Any]: 查询结果，包含：
                - query: 原始查询文本
                - intent: 识别出的查询意图
                - paths: 使用的查询路径列表
                - path_results: 每个路径的查询结果
                - merged_results: 合并后的结果统计
        """
        logger.info(f"开始智能图谱查询: '{query}', 实体: {entities}")
        
        # 1. 分析查询意图
        intent = self.analyze_query_intent(query)
        
        # 2. 决定查询路径
        paths = self.decide_query_paths(query, entities)
        
        # 3. 执行查询
        path_results = []
        for path in paths:
            results = self.execute_graph_query(path)
            path_results.append(results)
        
        # 4. 合并结果
        merged_results = self.merge_path_results(path_results)
        
        return {
            "query": query,
            "intent": intent.value,
            "paths": paths,
            "path_results": path_results,
            "merged_results": merged_results
        }
    
    def clear_cache(self):
        """
        清除所有缓存
        
        清除连接密度缓存、三元组缓存和节点映射缓存。
        应在图谱存储发生变化时调用此方法以确保数据一致性。
        """
        self.cache.clear()
        self._triplets_cache = None
        self._node_map_cache = None
        logger.info("已清除所有缓存")
