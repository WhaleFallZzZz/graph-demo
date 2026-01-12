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
    """连接密度信息"""
    
    def __init__(self, entity: str):
        self.entity = entity
        self.total_connections = 0
        self.connection_types = {}  # relation_type -> count
        self.neighbor_types = {}  # entity_type -> count
        self.neighbor_entities = []  # list of connected entities
        
    def add_connection(self, relation: str, neighbor_type: str, neighbor_entity: str):
        """添加连接信息"""
        self.total_connections += 1
        self.connection_types[relation] = self.connection_types.get(relation, 0) + 1
        self.neighbor_types[neighbor_type] = self.neighbor_types.get(neighbor_type, 0) + 1
        if neighbor_entity not in self.neighbor_entities:
            self.neighbor_entities.append(neighbor_entity)
    
    def get_dominant_relation(self) -> Optional[str]:
        """获取主导关系类型"""
        if not self.connection_types:
            return None
        return max(self.connection_types.items(), key=lambda x: x[1])[0]
    
    def get_dominant_neighbor_type(self) -> Optional[str]:
        """获取主导邻居类型"""
        if not self.neighbor_types:
            return None
        return max(self.neighbor_types.items(), key=lambda x: x[1])[0]


class GraphAgent:
    """智能图谱查询代理"""
    
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
    
    def __init__(self, graph_store, llm_instance=None, use_llm_classifier=True):
        """
        初始化 Graph-Agent
        
        Args:
            graph_store: 图谱存储对象
            llm_instance: LLM实例，用于意图分类
            use_llm_classifier: 是否使用LLM意图分类器，默认为True
        """
        self.graph_store = graph_store
        self.is_neo4j = "Neo4jPropertyGraphStore" in str(type(graph_store))
        self.cache = {}  # 缓存连接密度信息
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
        
        Args:
            query: 用户查询文本
            
        Returns:
            查询意图枚举
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
        
        Args:
            entity: 实体名称
            max_depth: 探测深度
            
        Returns:
            连接密度对象
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
        """探测 Neo4j 图谱的连接密度"""
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
        """探测内存图谱的连接密度"""
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
        
        Args:
            query: 用户查询文本
            entities: 查询实体列表
            
        Returns:
            查询路径列表，每个路径包含优先级和查询参数
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
        """决定综合查询的路径"""
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
        """根据特定意图决定路径"""
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
        """获取默认路径"""
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
        
        Args:
            path: 查询路径配置
            
        Returns:
            查询结果列表
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
        """执行 Neo4j 查询"""
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
        """执行内存图谱查询"""
        results = []
        entity = params.get('entity', '')
        relations = params.get('relations', [])
        max_depth = params.get('max_depth', 2)
        
        if not entity:
            return results
        
        try:
            # SimplePropertyGraphStore 使用 get_triplets() 获取所有三元组
            triplets = self.graph_store.get_triplets()
            
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
        
        Args:
            path_results: 多个路径的查询结果列表
            
        Returns:
            合并后的结果，包含路径统计和去重后的实体
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
        
        Args:
            query: 用户查询文本
            entities: 查询实体列表
            
        Returns:
            查询结果，包含意图、路径和合并结果
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
