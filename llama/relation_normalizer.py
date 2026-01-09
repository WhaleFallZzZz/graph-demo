"""
关系规范化模块
用于简化和规范化三元组中的关系描述
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


class RelationNormalizer:
    """关系规范化器 - 简化过长的关系描述，提取核心关系词"""
    
    # 标准关系类型映射（核心关系词）
    STANDARD_RELATIONS = {
        "导致", "引起", "造成", "产生",  # 因果关系
        "用于", "应用于", "用于治疗", "用于预防", "用于评估", "用于检测",  # 用途关系
        "包含", "属于", "包含于", "包括",  # 包含关系
        "表现为", "症状为", "显示为", "可见于",  # 表现关系
        "检查依据", "检查", "检测", "评估",  # 检查关系
        "量化关系", "量化", "测量",  # 量化关系
        "关联", "相关", "联系", "与...有关"  # 关联关系
    }
    
    # 关系简化规则：从复杂描述中提取核心关键词
    RELATION_SIMPLIFICATION_PATTERNS = [
        # 匹配"导致"类关系
        (r"影响.*形成.*导致", "导致"),
        (r"影响.*导致", "导致"),
        (r"影响.*形成", "导致"),
        (r".*导致.*", "导致"),
        (r".*引起.*", "导致"),
        (r".*造成.*", "导致"),
        
        # 匹配"用于"类关系
        (r".*用于.*治疗", "用于"),
        (r".*用于.*预防", "用于"),
        (r".*用于.*评估", "用于"),
        (r".*用于.*检测", "用于"),
        (r".*用于.*", "用于"),
        (r".*应用于.*", "用于"),
        
        # 匹配"表现为"类关系
        (r".*表现为.*", "表现为"),
        (r".*症状为.*", "表现为"),
        (r".*可见于.*", "表现为"),
        
        # 匹配"包含"类关系
        (r".*包含.*", "包含"),
        (r".*属于.*", "包含"),
        (r".*包括.*", "包含"),
        
        # 匹配"检查"类关系
        (r".*检查.*依据", "检查依据"),
        (r".*检查.*", "检查依据"),
        (r".*检测.*", "检查依据"),
        (r".*评估.*", "检查依据"),
        
        # 匹配"量化"类关系
        (r".*量化.*", "量化关系"),
        (r".*测量.*", "量化关系"),
        
        # 匹配"关联"类关系
        (r".*关联.*", "关联"),
        (r".*相关.*", "关联"),
        (r".*联系.*", "关联"),
    ]
    
    # 需要移除的冗余词汇（出现在关系描述中，但不属于核心关系）
    REDUNDANT_PATTERNS = [
        r"形成立体感的",
        r"差异性像差",
        r"视觉异常",
        r"病理性变化",
        r"如.*等",
        r"可能",
        r"会",
        r"可以",
        r"能够",
        r"往往",
        r"通常",
        r"一般",
    ]
    
    @classmethod
    def normalize(cls, relation: str, max_length: int = 10) -> str:
        """
        规范化关系描述，简化为核心关系词
        
        Args:
            relation: 原始关系描述
            max_length: 最大长度（字符数）
            
        Returns:
            规范化后的关系描述
        """
        if not relation:
            return "关联"
        
        original_relation = relation
        relation = str(relation).strip()
        
        # 1. 移除冗余词汇和描述性内容
        for pattern in cls.REDUNDANT_PATTERNS:
            relation = re.sub(pattern, "", relation, flags=re.IGNORECASE)
        
        # 2. 移除逗号后的内容（通常是详细说明）
        if "," in relation or "，" in relation:
            # 保留第一个主要部分
            relation = re.split(r'[,，]', relation)[0].strip()
        
        # 3. 移除括号内容
        relation = re.sub(r'[\(（].*?[\)）]', '', relation)
        
        # 4. 尝试匹配标准关系类型
        simplified = cls._extract_core_relation(relation)
        
        if simplified and simplified != relation:
            logger.debug(f"关系简化: '{original_relation}' -> '{simplified}'")
            relation = simplified
        
        # 5. 如果还是太长，尝试进一步提取关键词
        if len(relation) > max_length:
            relation = cls._extract_keywords(relation, max_length)
            if relation != original_relation:
                logger.info(f"关系过长，已简化: '{original_relation}' ({len(original_relation)}字) -> '{relation}' ({len(relation)}字)")
        
        # 6. 确保长度不超过限制
        if len(relation) > max_length:
            # 保留前max_length个字符，但尝试在词边界截断
            relation = cls._smart_truncate(relation, max_length)
            logger.warning(f"关系过长({len(original_relation)}字)，截断: '{original_relation}' -> '{relation}'")
        
        # 7. 确保不为空
        if not relation or len(relation.strip()) == 0:
            logger.warning(f"关系简化后为空，使用默认值: '{original_relation}'")
            return "关联"
        
        return relation.strip()
    
    @classmethod
    def _extract_core_relation(cls, relation: str) -> Optional[str]:
        """从关系描述中提取核心关系词"""
        relation_lower = relation.lower()
        
        # 检查是否直接匹配标准关系
        for std_rel in cls.STANDARD_RELATIONS:
            if std_rel in relation:
                return std_rel
        
        # 使用模式匹配提取
        for pattern, replacement in cls.RELATION_SIMPLIFICATION_PATTERNS:
            if re.search(pattern, relation, re.IGNORECASE):
                return replacement
        
        return None
    
    @classmethod
    def _extract_keywords(cls, relation: str, max_length: int) -> str:
        """从长描述中提取关键词"""
        # 优先提取动词（通常是关系的核心）
        verbs = ["导致", "引起", "造成", "用于", "包含", "表现为", "检查", "检测", "评估", "关联", "相关"]
        
        for verb in verbs:
            if verb in relation and len(verb) <= max_length:
                # 如果动词在关系中出现，优先使用动词
                return verb
        
        # 如果没有找到标准动词，尝试提取前几个字符的关键部分
        # 移除常见的前缀词
        prefixes = ["影响", "可能", "会", "可以", "能够", "往往", "通常", "一般"]
        for prefix in prefixes:
            if relation.startswith(prefix):
                relation = relation[len(prefix):].strip()
                break
        
        # 如果还是太长，直接截断
        if len(relation) > max_length:
            relation = relation[:max_length]
        
        return relation
    
    @classmethod
    def _smart_truncate(cls, text: str, max_length: int) -> str:
        """智能截断，尝试在词边界截断"""
        if len(text) <= max_length:
            return text
        
        # 尝试在标点符号处截断
        for punct in ["，", ",", "、", "；", ";", "。", "."]:
            idx = text.rfind(punct, 0, max_length)
            if idx > 0:
                return text[:idx].strip()
        
        # 尝试在常见连接词处截断
        for connector in ["的", "和", "或", "及"]:
            idx = text.rfind(connector, 0, max_length)
            if idx > 0 and idx < max_length - 2:  # 保留一些余量
                return text[:idx].strip()
        
        # 直接截断
        return text[:max_length].strip()
