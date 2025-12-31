"""
实体类型语义识别模块 - 完全依赖LLM语义分析，无任何限制
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class EntityTypeRules:
    """实体类型语义识别引擎 - 完全信任LLM语义分析"""
    
    @classmethod
    def validate_llm_entity_type(cls, llm_type: str, entity_name: str) -> str:
        """
        完全信任LLM的语义分析结果，不再进行任何限制
        只进行基本的格式清理
        """
        if not llm_type or not llm_type.strip():
            return "概念"
        
        # 完全信任LLM的语义分析，不再验证是否在预定义列表中
        # 只进行基本的格式清理
        return llm_type.strip()
    
    @classmethod
    def should_use_llm_semantic_analysis(cls, entity_name: str) -> bool:
        """总是使用LLM语义分析"""
        return True

# 更新EnhancedEntityExtractor中的方法
import enhanced_entity_extractor

# 替换原有的infer_entity_type方法，完全信任LLM语义分析
def new_infer_entity_type(self, entity_name: str, llm_type: str = None) -> str:
    """
    新的实体类型推断方法 - 完全信任LLM语义分析
    """
    if llm_type:
        # 完全信任LLM给出的类型，不再进行任何限制或验证
        return EntityTypeRules.validate_llm_entity_type(llm_type, entity_name)
    else:
        # 如果没有LLM类型，返回None表示需要LLM分析
        return None

# 应用新的方法
enhanced_entity_extractor.EnhancedEntityExtractor.infer_entity_type = new_infer_entity_type

print("✅ 实体类型推断已完全改为无限制的语义识别模式")