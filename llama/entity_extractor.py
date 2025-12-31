"""
增强的实体类型提取器 - 完全依赖LLM语义分析，无任何限制
"""

import logging
from typing import List, Tuple, Dict, Any, Optional
import json
import re

# Import EntityNode and Relation from llama_index.core
from llama_index.core.graph_stores.types import EntityNode, Relation

logger = logging.getLogger(__name__)

try:
    from utils import parse_llm_output_with_types, safe_json_parse
except (ImportError, ModuleNotFoundError):
    logger.warning("enhanced_utils.py not found. Using basic fallback.")
    def parse_llm_output_with_types(llm_output: str) -> List[Dict[str, str]]:
        # 本地简易回退
        import re
        results = []
        pattern = r'"head"\s*:\s*"(.*?)"\s*,\s*"head_type"\s*:\s*"(.*?)"\s*,\s*"relation"\s*:\s*"(.*?)"\s*,\s*"tail"\s*:\s*"(.*?)"\s*,\s*"tail_type"\s*:\s*"(.*?)"'
        matches = re.findall(pattern, llm_output, re.DOTALL)
        for h, ht, r, t, tt in matches:
             results.append({"head":h, "head_type":ht, "relation":r, "tail":t, "tail_type":tt})
        return results

class EnhancedEntityExtractor:
    """增强的实体提取器 - 完全信任LLM语义分析"""
    
    @classmethod
    def extract_enhanced_triplets(cls, llm_output: str) -> List[Dict[str, Any]]:
        """提取增强的三元组，完全信任LLM的语义分析结果"""
        enhanced_triplets = []
        
        # 添加调试日志以查看LLM原始输出
        logger.info(f"LLM原始输出 (长度: {len(llm_output)}): {llm_output[:500]}...")
        
        # 使用 enhanced_utils 中的 parse_llm_output_with_types 
        # 这个函数已经集成了 safe_json_parse 和带类型的正则回退
        parsed_dicts = parse_llm_output_with_types(llm_output)
        
        if parsed_dicts:
            for item in parsed_dicts:
                head = item.get("head", "").strip()
                head_type = item.get("head_type", "").strip()
                relation = item.get("relation", "").strip()
                tail = item.get("tail", "").strip()
                tail_type = item.get("tail_type", "").strip()
                
                # 只有当head, relation, tail都存在且不全是标点符号时才添加
                if head and relation and tail:
                    # 避免尾部是逗号等标点符号的无效提取
                    if tail in {",", ".", "。", "，", "、"}:
                         logger.warning(f"检测到无效的尾部实体(标点符号): '{tail}'，跳过该三元组")
                         continue

                    enhanced_triplets.append({
                        "head": head,
                        "head_type": head_type or "概念",
                        "relation": relation,
                        "tail": tail,
                        "tail_type": tail_type or "概念"
                    })
                    
                    logger.debug(f"提取LLM语义三元组: {head}({head_type}) - {relation} - {tail}({tail_type})")
        
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

# 修改 parse_llm_output_to_enhanced_triplets 函数以返回 EntityNode, Relation 对象
def parse_llm_output_to_enhanced_triplets(llm_output: str) -> List[Tuple[EntityNode, Relation, EntityNode]]:
    """增强的解析函数，完全信任LLM的语义分析结果"""
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
            # 清理名称
            head_name = str(head_name).strip()
            tail_name = str(tail_name).strip()
            relation_type = str(relation_type).strip()
            
            # 验证：跳过纯标点或空的实体/关系
            invalid_symbols = {",", ".", "。", "，", "、", " ", "\\", "/", ";", ":", "?", "!", "'", "\"", "(", ")", "[", "]", "{", "}", "-", "_", "+", "=", "*", "&", "^", "%", "$", "#", "@", "~", "`", "<", ">", "|"}
            
            def is_invalid(text):
                if not text: return True
                if text in invalid_symbols: return True
                return all(char in invalid_symbols for char in text)

            if is_invalid(head_name) or is_invalid(tail_name) or is_invalid(relation_type):
                logger.warning(f"跳过无效实体/关系: '{head_name}' - '{relation_type}' - '{tail_name}'")
                continue

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

# 保持原有的函数名兼容性
parse_dynamic_triplets = parse_llm_output_to_enhanced_triplets