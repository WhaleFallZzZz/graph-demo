"""
Neo4j 文本清理工具
在存入Neo4j前清理和转义特殊字符，确保数据完整性和查询稳定性
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class Neo4jTextSanitizer:
    """Neo4j文本清理器 - 处理实体名称和关系标签中的特殊字符"""
    
    # Neo4j Cypher 中有特殊含义的字符
    SPECIAL_CHARS = {
        "'": "'",      # 单引号 -> 右单引号
        '"': '"',      # 双引号 -> 右双引号
        '`': 'ˋ',      # 反引号
        '\\': '＼',    # 反斜杠
        ':': '：',     # 半角冒号 -> 全角冒号
        '：': '：',     # 全角冒号保持不变(已经是全角)
        '*': '＊',     # 星号
        '?': '？',     # 问号
        '[': '［',     # 左方括号
        ']': '］',     # 右方括号
        '{': '｛',     # 左花括号
        '}': '｝',     # 右花括号
        '|': '｜',     # 竖线
        ';': '；',     # 分号
        '(': '（',     # 左圆括号转全角（用于节点名）
        ')': '）',     # 右圆括号转全角（用于节点名）
    }
        
    # 关系标签中需要完全移除的字符（不转换为全角）
    RELATION_REMOVE_CHARS = {
        "'": '',       # 单引号直接移除
        '"': '',       # 双引号直接移除
        '`': '',       # 反引号直接移除
        '(': '',       # 左圆括号移除（保持关系标签简洁）
        ')': '',       # 右圆括号移除
        '\\': '',      # 反斜杠移除
        ':': '',       # 半角冒号移除
        '：': '',      # 全角冒号移除
        '*': '',       # 星号移除
    }
    
    # 需要完全移除的字符(通常是控制字符)
    REMOVE_CHARS = {
        '\n': '',      # 换行符
        '\r': '',      # 回车符
        '\t': ' ',     # 制表符转为空格
        '\x00': '',    # 空字符
        '\ufffd': '',  # 替换字符(无效Unicode)
    }
    
    # 危险的Cypher关键字(需要特别处理)
    CYPHER_KEYWORDS = {
        'MATCH', 'WHERE', 'RETURN', 'CREATE', 'MERGE', 'DELETE', 'DETACH',
        'SET', 'REMOVE', 'WITH', 'UNION', 'UNWIND', 'ORDER', 'SKIP', 'LIMIT',
        'OPTIONAL', 'CALL', 'YIELD', 'FOREACH', 'CASE', 'WHEN', 'THEN', 'ELSE',
        'END', 'AND', 'OR', 'XOR', 'NOT', 'IN', 'STARTS', 'ENDS', 'CONTAINS',
        'IS', 'NULL', 'TRUE', 'FALSE', 'AS', 'DISTINCT', 'ASC', 'DESC',
    }
    
    @classmethod
    def sanitize_node_name(cls, text: str, max_length: int = 200) -> str:
        """
        清理节点名称
        
        Args:
            text: 原始文本
            max_length: 最大长度限制
            
        Returns:
            清理后的文本
        """
        if not text:
            return ""
        
        # 1. 转换为字符串并去除首尾空格
        text = str(text).strip()
        
        # 2. 移除控制字符
        for char, replacement in cls.REMOVE_CHARS.items():
            text = text.replace(char, replacement)
        
        # 3. 先移除单双引号和反引号（保持名称简洁）
        text = text.replace("'", '').replace('"', '').replace('`', '')
        
        # 4. 替换其他特殊字符为全角字符(保持语义)
        for char, replacement in cls.SPECIAL_CHARS.items():
            # 跳过已处理的引号
            if char not in ["'", '"', '`']:
                text = text.replace(char, replacement)
        
        # 5. 多个空格合并为一个
        text = re.sub(r'\s+', ' ', text)
        
        # 6. 去除首尾空格
        text = text.strip()
        
        # 7. 检查是否是Cypher关键字,如果是则添加前缀
        if text.upper() in cls.CYPHER_KEYWORDS:
            text = f"Entity_{text}"
            logger.warning(f"节点名称是Cypher关键字，添加前缀: {text}")
        
        # 8. 长度限制
        if len(text) > max_length:
            original_length = len(text)
            text = text[:max_length]
            logger.warning(f"节点名称过长({original_length}字符)，截断到{max_length}字符: {text}...")
        
        # 9. 确保不为空
        if not text:
            logger.error("清理后节点名称为空，使用默认值")
            return "未命名实体"
        
        return text
    
    @classmethod
    def sanitize_relation_label(cls, text: str, max_length: int = 100) -> str:
        """
        清理关系标签
        关系标签需要更严格的清理，直接移除引号等字符
        
        Args:
            text: 原始关系标签
            max_length: 最大长度
            
        Returns:
            清理后的关系标签
        """
        if not text:
            return "相关"
        
        # 1. 基本清理
        text = str(text).strip()
        
        # 2. 移除控制字符
        for char, replacement in cls.REMOVE_CHARS.items():
            text = text.replace(char, replacement)
        
        # 3. 优先使用关系标签专用的移除规则（直接删除引号等）
        for char, replacement in cls.RELATION_REMOVE_CHARS.items():
            text = text.replace(char, replacement)
        
        # 4. 其他特殊字符替换为全角
        for char, replacement in cls.SPECIAL_CHARS.items():
            # 跳过已经处理的字符
            if char not in cls.RELATION_REMOVE_CHARS:
                text = text.replace(char, replacement)
        
        # 5. 多个空格合并
        text = re.sub(r'\s+', ' ', text)
        
        # 6. 去除首尾空格
        text = text.strip()
        
        # 7. 检查Cypher关键字
        if text.upper() in cls.CYPHER_KEYWORDS:
            text = f"REL_{text}"
            logger.warning(f"关系标签是Cypher关键字，添加前缀: {text}")
        
        # 8. 长度限制
        if len(text) > max_length:
            original_length = len(text)
            text = text[:max_length]
            logger.warning(f"关系标签过长({original_length}字符)，截断到{max_length}字符: {text}...")
        
        # 9. 确保不为空
        if not text:
            logger.error("清理后关系标签为空，使用默认值")
            return "相关"
        
        return text
    
    @classmethod
    def sanitize_entity_type(cls, text: str) -> str:
        """
        清理实体类型(Label)
        要求更严格,因为Neo4j的Label有更多限制
        
        Args:
            text: 原始实体类型
            
        Returns:
            清理后的实体类型
        """
        if not text:
            return "Entity"
        
        # 1. 基本清理
        text = str(text).strip()
        
        # 2. 移除所有特殊字符和空格
        # Label只能包含字母、数字、下划线、汉字
        # 保留汉字、字母、数字、下划线
        text = re.sub(r'[^\w\u4e00-\u9fff]', '_', text)
        
        # 3. 不能以数字开头
        if text and text[0].isdigit():
            text = f"Type_{text}"
        
        # 4. 多个下划线合并为一个
        text = re.sub(r'_+', '_', text)
        
        # 5. 去除首尾下划线
        text = text.strip('_')
        
        # 5.5 移除末尾的下划线+数字模式（如 "疾病_12" -> "疾病"）
        # 这通常是LLM输出带来的额外信息
        text = re.sub(r'_\d+$', '', text)
        
        # 6. 再次去除首尾下划线（防止移除数字后留下下划线）
        text = text.strip('_')
        
        # 7. 检查Cypher关键字
        if text.upper() in cls.CYPHER_KEYWORDS:
            text = f"Type_{text}"
        
        # 8. 确保不为空
        if not text:
            logger.error("清理后实体类型为空，使用默认值")
            return "Entity"
        
        return text
    
    @classmethod
    def validate_text(cls, text: str, field_name: str = "text") -> bool:
        """
        验证文本是否安全(不含危险字符)
        
        Args:
            text: 待验证的文本
            field_name: 字段名称(用于日志)
            
        Returns:
            True表示安全,False表示不安全
        """
        if not text:
            return True
        
        # 检查是否包含SQL注入相关的危险模式
        dangerous_patterns = [
            r';\s*DROP\s+',
            r';\s*DELETE\s+',
            r';\s*CREATE\s+',
            r'EXEC\s*\(',
            r'EXECUTE\s*\(',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                logger.error(f"检测到危险模式在 {field_name}: {text[:50]}")
                return False
        
        return True
    
    @classmethod
    def batch_sanitize(cls, items: dict) -> dict:
        """
        批量清理三元组数据
        
        Args:
            items: 包含 head, head_type, relation, tail, tail_type 的字典
            
        Returns:
            清理后的字典
        """
        sanitized = {}
        
        if 'head' in items:
            sanitized['head'] = cls.sanitize_node_name(items['head'])
        
        if 'head_type' in items:
            sanitized['head_type'] = cls.sanitize_entity_type(items['head_type'])
        
        if 'relation' in items:
            sanitized['relation'] = cls.sanitize_relation_label(items['relation'])
        
        if 'tail' in items:
            sanitized['tail'] = cls.sanitize_node_name(items['tail'])
        
        if 'tail_type' in items:
            sanitized['tail_type'] = cls.sanitize_entity_type(items['tail_type'])
        
        return sanitized


# 便捷函数
def sanitize_for_neo4j(text: str, text_type: str = "name") -> str:
    """
    便捷函数：根据类型清理文本
    
    Args:
        text: 待清理的文本
        text_type: 文本类型 ("name", "label", "type")
        
    Returns:
        清理后的文本
    """
    if text_type == "name":
        return Neo4jTextSanitizer.sanitize_node_name(text)
    elif text_type == "label":
        return Neo4jTextSanitizer.sanitize_relation_label(text)
    elif text_type == "type":
        return Neo4jTextSanitizer.sanitize_entity_type(text)
    else:
        logger.warning(f"未知的文本类型: {text_type}，使用默认清理")
        return Neo4jTextSanitizer.sanitize_node_name(text)
