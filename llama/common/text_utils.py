"""
知识图谱系统的文本处理工具。

本模块提供统一的文本处理函数，包括：
- 文本清理和规范化
- Neo4j 安全文本格式化
"""

import re
import logging

logger = logging.getLogger(__name__)

# Neo4j special characters that need to be escaped
# Note: Order matters - backslash must be replaced first to avoid double-escaping
NEO4J_SPECIAL_CHARS = [
    ('\\', '\\\\'),
    ('\'', "\\'"),
    ('"', '\\"'),
    ('\n', '\\n'),
    ('\r', '\\r'),
    ('\t', '\\t'),
    ('\b', '\\b'),
    ('\f', '\\f')
]

# Characters to remove for cleaner text
CLEAN_CHARS = r'[_`""\'""]'


def clean_text(text: str, remove_special: bool = True) -> str:
    """
    通过删除不需要的字符来清理和规范化文本。
    
    删除特殊字符（如下划线、引号），并将多个空格规范化为单个空格。
    主要用于清理实体名称、关系类型等文本数据。
    
    Args:
        text: 要清理的输入文本
        remove_special: 是否删除特殊字符，包括下划线、引号等（默认：True）
        
    Returns:
        清理后的文本
        
    Raises:
        TypeError: 如果输入不是字符串类型
        
    使用示例：
        >>> clean_text("_test_text_")
        'testtext'
        
        >>> clean_text("  test  ")
        'test'
        
        >>> clean_text("test_text", remove_special=False)
        'test_text'
    """
    if not text:
        return text
    
    if not isinstance(text, str):
        raise TypeError(f"Expected string, got {type(text)}")
    
    result = text.strip()
    
    if remove_special:
        # 删除特殊字符但保留空格
        result = re.sub(CLEAN_CHARS, "", result)
    
    # 将多个空格规范化为单个空格
    result = re.sub(r'\s+', ' ', result)
    
    return result.strip()


def sanitize_for_neo4j(text: str, max_length: int = 1000) -> str:
    """
    清理文本以便在 Neo4j 查询中安全使用。
    
    转义 Neo4j Cypher 查询中的特殊字符（如单引号、双引号、反斜杠等），
    并限制文本长度以防止注入攻击和查询错误。
    
    Args:
        text: 要清理的输入文本
        max_length: 允许的最大长度，超过此长度将被截断（默认：1000）
        
    Returns:
        对 Neo4j 安全的清理后文本，特殊字符已转义
        
    Raises:
        TypeError: 如果输入不是字符串类型
        
    使用示例：
        >>> sanitize_for_neo4j("test's name")
        "test\\'s name"
        
        >>> sanitize_for_neo4j("test\"quote")
        'test\\"quote'
        
        >>> sanitize_for_neo4j("test\nnewline")
        'test\\nnewline'
        
        >>> sanitize_for_neo4j("a" * 2000, max_length=100)
        'aaaaaaaaaa...'  # 截断到 100 个字符
    """
    if not text:
        return text
    
    if not isinstance(text, str):
        raise TypeError(f"Expected string, got {type(text)}")
    
    # 如果太长则截断
    if len(text) > max_length:
        logger.warning(f"Text truncated from {len(text)} to {max_length} characters")
        text = text[:max_length]
    
    # 转义特殊字符（按顺序处理，避免重复转义）
    result = text
    for char, escaped in NEO4J_SPECIAL_CHARS:
        result = result.replace(char, escaped)
    
    return result
