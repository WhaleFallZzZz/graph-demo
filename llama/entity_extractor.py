"""
增强的实体类型提取器 - 完全依赖LLM语义分析，无任何限制
"""

import logging
from typing import List, Tuple, Dict, Any, Optional
import json
import queue
import threading
import resource
import time
import sys
import asyncio
import re
from concurrent.futures import ThreadPoolExecutor

# Import EntityNode and Relation from llama_index.core
from llama_index.core.graph_stores.types import EntityNode, Relation
from llama_index.core.schema import BaseNode
from llama_index.core.indices.property_graph import DynamicLLMPathExtractor

# 导入 common 模块的工具
from llama.common import (
    safe_json_parse,
    parse_llm_output,
    clean_text,
    sanitize_for_neo4j,
    DynamicThreadPool,
    retry_on_failure
)
from llama.config import RERANK_CONFIG, EXTRACTOR_CONFIG
from llama.custom_siliconflow_rerank import CustomSiliconFlowRerank
from llama_index.core.schema import TextNode, NodeWithScore, QueryBundle

logger = logging.getLogger(__name__)

ENTITY_TYPE_SCHEMA: Dict[str, str] = {}

# 注释 StandardTermMapper (标准词映射) 相关代码
# try:
#     from enhanced_entity_extractor import StandardTermMapper
# except ImportError:
#     # Fallback if file not found or circular import
#     logger.warning("StandardTermMapper not found in enhanced_entity_extractor.py")
#     class StandardTermMapper:
#         @classmethod
#         def process_triplets(cls, triplets):
#             return triplets

class EnhancedEntityExtractor:
    """增强的实体提取器 - 完全信任LLM语义分析"""
    
    @classmethod
    def extract_enhanced_triplets(cls, llm_output: str) -> List[Dict[str, Any]]:
        enhanced_triplets = []
        
       # --- 新增：清洗 R1 模型的思考过程 ---
        # 1. 去除 <think> 标签及内容
        llm_output = re.sub(r'<think>.*?</think>', '', llm_output, flags=re.DOTALL)
        
        # 2. 去除 Markdown 代码块标记 (```json ... ```)
        llm_output = re.sub(r'^```json\s*', '', llm_output, flags=re.MULTILINE)
        llm_output = re.sub(r'^```\s*', '', llm_output, flags=re.MULTILINE)
        
        # 3. 清理首尾空白
        llm_output = llm_output.strip()
        # -----------------------------------
        try:
            start = llm_output.find('[')
            end = llm_output.rfind(']')
            if start != -1 and end != -1:
                json_str = llm_output[start : end + 1]
                # 尝试直接解析
                return json.loads(json_str)
        except Exception:
            pass
        
        logger.info(f"清洗后的 LLM 输出: {llm_output[:200]}...") # 调试日志
        
        # 使用 common 模块中的 parse_llm_output
        parsed_dicts = parse_llm_output(llm_output)
        
        if parsed_dicts:
            for item in parsed_dicts:
                head = (item.get("head") or "").strip()
                head_type = (item.get("head_type") or "").strip()
                relation = (item.get("relation") or "").strip()
                tail = (item.get("tail") or "").strip()
                tail_type = (item.get("tail_type") or "").strip()
                
                # 只有当head, relation, tail都存在且不全是标点符号时才添加
                if head and relation and tail:
                    # 避免尾部是逗号等标点符号的无效提取
                    if tail in {",", ".", "。", "，", "、"}:
                         logger.warning(f"检测到无效的尾部实体(标点符号): '{tail}'，跳过该三元组")
                         continue
                    
                    # 确保实体类型和关系不为空或 None
                    head_type = head_type.strip() if head_type else "概念"
                    tail_type = tail_type.strip() if tail_type else "概念"
                    relation = relation.strip() if relation else None
                    
                    # 如果关系为空，跳过该三元组
                    if not relation:
                        logger.warning(f"关系类型为空，跳过三元组: {head} - {relation} - {tail}")
                        continue

                    enhanced_triplets.append({
                        "head": head,
                        "head_type": head_type,
                        "relation": relation,
                        "tail": tail,
                        "tail_type": tail_type
                    })
                    
                    logger.debug(f"提取LLM语义三元组: {head}({head_type}) - {relation} - {tail}({tail_type})")
        
        # 应用术语映射标准化 先注释
        # enhanced_triplets = StandardTermMapper.process_triplets(enhanced_triplets)
        
        if not enhanced_triplets:
            logger.warning("未能从LLM输出中提取到任何有效的三元组")
             
        return enhanced_triplets
    
    @classmethod
    def validate_llm_entity_types(cls, enhanced_triplets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """验证LLM返回的实体类型 - 完全信任LLM，不再进行任何限制"""
        # 完全信任LLM的语义分析，不再验证类型是否在预定义列表中
        # 只进行基本的格式清理
        validated_triplets = []
        for triplet in enhanced_triplets:
            # 只进行基本的非空检查，完全信任LLM的语义判断
            head_type = triplet.get("head_type", "概念")
            tail_type = triplet.get("tail_type", "概念")
            
            # 只清理空白字符，不再进行任何类型限制
            triplet["head_type"] = head_type.strip() if head_type else "概念"
            triplet["tail_type"] = tail_type.strip() if tail_type else "概念"
            
            validated_triplets.append(triplet)
        
        return validated_triplets


def validate_and_map_entity_type(entity_type: str) -> Optional[str]:
    """
    验证和映射实体类型到标准本体
    
    Args:
        entity_type: LLM 返回的实体类型
    
    Returns:
        标准化后的实体类型，如果不在允许列表中返回 None 或 "Concept"
    
    规则：
    - 如果在 ENTITY_TYPE_SCHEMA 中，返回映射后的标准类型
    - 如果不在，根据配置决定：映射为 "Concept" 或返回 None（丢弃）
    """
    if not entity_type:
        return None
    
    # 清理类型字符串
    entity_type = entity_type.strip()
    
    # 直接匹配
    if entity_type in ENTITY_TYPE_SCHEMA:
        return ENTITY_TYPE_SCHEMA[entity_type]

    # 模糊匹配：检查是否包含标准类型关键词
    standard_types = [
        "疾病", "部位", "治疗", "症状","体征", "检查",  # 原有临床类
        "流行病学方法", "统计指标", "研究类型",    # 新增科研类
        "卫生经济学", "公共卫生策略", "行动计划",  # 新增公卫类
        "风险因素", "设备","异常","部位","生理","诊疗","干预","风险","因素"                       # 新增其他类
    ]
    for std_type in standard_types:
        if std_type in entity_type:
            return std_type
    
    # 不在允许列表中，根据配置决定
    if EXTRACTOR_CONFIG.get("strict_entity_type_schema", True):
        action = EXTRACTOR_CONFIG.get("invalid_entity_type_action", "map_to_concept")
        if action == "discard":
            return None  # 返回 None 表示应该丢弃
        else:  # map_to_concept
            # return "Concept"  # 映射为 Concept
            return None
    
    # 如果没有启用严格模式，返回原值（向后兼容）
    return entity_type


# 修改 parse_llm_output_to_enhanced_triplets 函数以返回 EntityNode, Relation 对象
def parse_llm_output_to_enhanced_triplets(llm_output: str) -> List[Tuple[EntityNode, Relation, EntityNode]]:
    """增强的解析函数，完全信任LLM的语义分析结果，并清理特殊字符"""
    from llama.neo4j_text_sanitizer import Neo4jTextSanitizer
    
    enhanced_triplets_dicts = EnhancedEntityExtractor.extract_enhanced_triplets(llm_output)
    
    # 验证LLM返回的实体类型 - 完全信任模式
    validated_triplets = EnhancedEntityExtractor.validate_llm_entity_types(enhanced_triplets_dicts)
    
    result_triplets = []
    for triplet_dict in validated_triplets:
        head_name = triplet_dict.get("head", "")
        head_type = triplet_dict.get("head_type", "概念")
        relation_type = triplet_dict.get("relation", "关联")
        tail_name = triplet_dict.get("tail", "")
        tail_type = triplet_dict.get("tail_type", "概念")
        
        if head_name and relation_type and tail_name:
            # 使用 common 模块中的 clean_text 进行基本清理
            head_name = clean_text(head_name, remove_special=False)
            tail_name = clean_text(tail_name, remove_special=False)
            relation_type = clean_text(relation_type, remove_special=False)

            # ---------------------------------------------------------
            # 强制映射层 (Standardization Error Fix) - 用户请求的强校验钩子
            # 已注释：移除 StandardTermMapper (标准词映射)
            # ---------------------------------------------------------
            # try:
            #     # 再次尝试标准化，确保在创建节点前强制应用标准术语
            #     std_head = StandardTermMapper.standardize(head_name)
            #     if std_head in StandardTermMapper.STANDARD_ENTITIES:
            #         if head_name != std_head:
            #             logger.info(f"🔧 强制纠偏 (Head): {head_name} -> {std_head}")
            #         head_name = std_head
            #     
            #     std_tail = StandardTermMapper.standardize(tail_name)
            #     if std_tail in StandardTermMapper.STANDARD_ENTITIES:
            #         if tail_name != std_tail:
            #             logger.info(f"🔧 强制纠偏 (Tail): {tail_name} -> {std_tail}")
            #         tail_name = std_tail
            # except Exception as e:
            #     logger.warning(f"StandardTermMapper 强校验失败: {e}")
            # ---------------------------------------------------------
            
            # 验证：跳过纯标点或空的实体/关系
            invalid_symbols = {",", ".", "。", "，", "、", " ", "\\", "/", ";", ":", "?", "!", "'", "\"", "(", ")", "[", "]", "{", "}", "-", "_", "+", "=", "*", "&", "^", "%", "$", "#", "@", "~", "`", "<", ">", "|"}
            
            def is_invalid(text):
                if not text: return True
                if text in invalid_symbols: return True
                return all(char in invalid_symbols for char in text)

            if is_invalid(head_name) or is_invalid(tail_name) or is_invalid(relation_type):
                logger.warning(f"跳过无效实体/关系: '{head_name}' - '{relation_type}' - '{tail_name}'")
                continue
            
            # 使用 common 模块中的 sanitize_for_neo4j 进行 Neo4j 安全清理
            # 记录清理前的值(用于日志对比)
            original_head = head_name
            original_tail = tail_name
            original_relation = relation_type
            original_head_type = head_type
            original_tail_type = tail_type
            
            # 清理节点名称
            head_name = Neo4jTextSanitizer.sanitize_node_name(head_name)
            tail_name = Neo4jTextSanitizer.sanitize_node_name(tail_name)
            
            # 清理关系标签（包含关系规范化）
            original_relation_length = len(relation_type)
            relation_type = Neo4jTextSanitizer.sanitize_relation_label(relation_type, max_length=10)
            
            # ===== 实体类型本体约束验证 =====
            # 清理实体类型(Label) - 确保有默认值
            head_type = head_type or "概念"
            tail_type = tail_type or "概念"
            head_type = Neo4jTextSanitizer.sanitize_entity_type(head_type)
            tail_type = Neo4jTextSanitizer.sanitize_entity_type(tail_type)
            
            # 应用本体约束：验证和映射实体类型
            original_head_type = head_type
            original_tail_type = tail_type
            head_type = validate_and_map_entity_type(head_type)
            tail_type = validate_and_map_entity_type(tail_type)
            
            # 如果类型被丢弃（返回 None），跳过该三元组
            if head_type is None or tail_type is None:
                discarded_types = []
                if head_type is None:
                    discarded_types.append(f"head: {original_head_type}")
                if tail_type is None:
                    discarded_types.append(f"tail: {original_tail_type}")
                logger.warning(
                    f"实体类型不在允许列表中，跳过三元组: "
                    f"{head_name}({original_head_type}) - {relation_type} - {tail_name}({original_tail_type}) | "
                    f"丢弃原因: {', '.join(discarded_types)}"
                )
                continue
            
            # 如果类型被映射为 Concept，记录日志
            if original_head_type != head_type and head_type == "Concept":
                logger.info(f"实体类型映射: {head_name} '{original_head_type}' -> 'Concept'")
            if original_tail_type != tail_type and tail_type == "Concept":
                logger.info(f"实体类型映射: {tail_name} '{original_tail_type}' -> 'Concept'")
            
            # 确保清理后不为 None 或空字符串（备用检查）
            if not head_type or head_type == "None":
                head_type = "Concept"
                logger.warning(f"实体类型为空，使用默认值 'Concept': {head_name}")
            if not tail_type or tail_type == "None":
                tail_type = "Concept"
                logger.warning(f"实体类型为空，使用默认值 'Concept': {tail_name}")
            # ===== 实体类型验证结束 =====
            
            # 验证关系类型不为空
            if not relation_type or relation_type == "None":
                logger.warning(f"关系类型为空，跳过三元组: {head_name} - {relation_type} - {tail_name}")
                continue
            
            # 如果清理后发生了变化，记录日志
            relation_changed = original_relation != relation_type
            relation_simplified = original_relation_length > 10 and len(relation_type) <= 10
            
            if (original_head != head_name or original_tail != tail_name or 
                relation_changed or original_head_type != head_type or 
                original_tail_type != tail_type):
                if relation_simplified:
                    logger.info(
                        f"🔧 关系简化: "
                        f"[{original_relation}] ({original_relation_length}字) -> [{relation_type}] ({len(relation_type)}字) | "
                        f"三元组: {head_name} - {tail_name}"
                    )
                else:
                    logger.debug(
                        f"🧹 字符清理: "
                        f"[{original_head}({original_head_type})] -> [{head_name}({head_type})], "
                        f"[{original_relation}] -> [{relation_type}], "
                        f"[{original_tail}({original_tail_type})] -> [{tail_name}({tail_type})]"
                    )
            
            # 再次验证清理后不为空
            if not head_name or not tail_name or not relation_type:
                logger.error(f"清理后实体/关系为空，跳过: {head_name} - {relation_type} - {tail_name}")
                continue
            # ===== 清理结束 =====

            logger.info(f"创建语义三元组: {head_name}({head_type}) - {relation_type} - {tail_name}({tail_type})")
                
            head_node = EntityNode(name=head_name, label=head_type)
            tail_node = EntityNode(name=tail_name, label=tail_type)
            
            relation = Relation(
                source_id=head_node.id,
                target_id=tail_node.id,
                label=relation_type
            )
            result_triplets.append((head_node, relation, tail_node))
        else:
            logger.warning(f"跳过无效三元组: {triplet_dict}")
            
    return result_triplets


def parse_dynamic_triplets(llm_output: str) -> List[Tuple[EntityNode, Relation, EntityNode]]:
    return parse_llm_output_to_enhanced_triplets(llm_output)
