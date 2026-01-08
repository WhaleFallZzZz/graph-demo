"""
知识图谱系统的文本处理工具。

本模块提供统一的文本处理函数，包括：
- 文本清理和规范化
- Neo4j 安全文本格式化
- 代码块提取
- 特殊字符处理
- 空白字符规范化
"""

import re
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# Neo4j special characters that need to be escaped
NEO4J_SPECIAL_CHARS = {
    "'": "\\'",
    '"': '\\"',
    '\\': '\\\\',
    '\n': '\\n',
    '\r': '\\r',
    '\t': '\\t',
    '\b': '\\b',
    '\f': '\\f'
}

# Characters to remove for cleaner text
CLEAN_CHARS = r'[_`""\'""]'


def clean_text(text: str, remove_special: bool = True) -> str:
    """
    通过删除不需要的字符来清理和规范化文本。
    
    Args:
        text: 要清理的输入文本
        remove_special: 是否删除特殊字符，如下划线、引号
        
    Returns:
        清理后的文本
        
    Examples:
        >>> clean_text("_test_text_")
        'test text'
        >>> clean_text("test", remove_special=False)
        'test'
    """
    if not text:
        return text
    
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
    
    转义特殊字符并限制长度以防止注入攻击。
    
    Args:
        text: 要清理的输入文本
        max_length: 允许的最大长度（默认：1000）
        
    Returns:
        对 Neo4j 安全的清理后文本
        
    Examples:
        >>> sanitize_for_neo4j("test's name")
        "test\\'s name"
    """
    if not text:
        return text
    
    # 如果太长则截断
    if len(text) > max_length:
        logger.warning(f"Text truncated from {len(text)} to {max_length} characters")
        text = text[:max_length]
    
    # 转义特殊字符
    result = text
    for char, escaped in NEO4J_SPECIAL_CHARS.items():
        result = result.replace(char, escaped)
    
    return result


def normalize_whitespace(text: str) -> str:
    """
    规范化文本中的空白字符。
    
    将多个空格/制表符/换行符合并为单个空格。
    
    Args:
        text: 包含不规则空白字符的输入文本
        
    Returns:
        空白字符已规范化的文本
        
    Examples:
        >>> normalize_whitespace("test  multiple   spaces")
        'test multiple spaces'
    """
    if not text:
        return text
    
    # 将所有空白字符序列替换为单个空格
    result = re.sub(r'\s+', ' ', text)
    
    return result.strip()


def remove_special_chars(text: str, keep_chars: str = '') -> str:
    """
    从文本中删除特殊字符，可选择保留某些字符。
    
    Args:
        text: 输入文本
        keep_chars: 要保留的字符（例如 '()-.')
        
    Returns:
        已删除特殊字符的文本
        
    Examples:
        >>> remove_special_chars("test@123!")
        'test123'
        >>> remove_special_chars("test(123)", keep_chars='()')
        'test(123)'
    """
    if not text:
        return text
    
    # 构建模式以保留字母数字和指定字符
    pattern = f'[^a-zA-Z0-9\\s{re.escape(keep_chars)}]'
    result = re.sub(pattern, '', text)
    
    return result


def extract_code_blocks(text: str, languages: Optional[List[str]] = None) -> List[str]:
    """
    从 markdown 格式的文本中提取代码块。
    
    Args:
        text: 包含代码块的输入文本
        languages: 可选的语言过滤器列表（例如 ['cypher', 'python']）
                  如果为 None，则提取所有代码块
        
    Returns:
        提取的代码块列表（不包含 markdown 标记）
        
    Examples:
        >>> extract_code_blocks("```python\\nprint('hello')\\n```")
        ["print('hello')"]
        >>> extract_code_blocks("```cypher\\nMATCH (n)\\n```", languages=['cypher'])
        ['MATCH (n)']
    """
    if not text:
        return []
    
    # 构建语言模式
    if languages:
        lang_pattern = '|'.join(re.escape(lang) for lang in languages)
        pattern = rf"```(?:{lang_pattern})?\s*(.*?)\s*```"
    else:
        pattern = r"```(?:\w+)?\s*(.*?)\s*```"
    
    # 提取所有代码块
    code_blocks = re.findall(pattern, text, re.DOTALL)
    
    return code_blocks


def remove_think_tags(text: str) -> str:
    """
    删除 DeepSeek think 标签及其内容。
    
    处理正确闭合和未闭合的标签。
    
    Args:
        text: 包含 think 标签的输入文本
        
    Returns:
        已删除 think 标签的文本
        
    Examples:
        >>> remove_think_tags("test <think>reasoning</think> result")
        'test  result'
    """
    if not text:
        return text
    
    # 删除正确闭合的标签
    result = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # 删除未闭合的标签（从 <think> 到结尾）
    result = re.sub(r'<think>.*', '', result, flags=re.DOTALL | re.IGNORECASE)
    
    # 清理多余的空白字符
    result = normalize_whitespace(result)
    
    return result


def truncate_text(text: str, max_length: int, suffix: str = '...') -> str:
    """
    将文本截断到最大长度，如果截断则添加后缀。
    
    Args:
        text: 输入文本
        max_length: 允许的最大长度
        suffix: 如果截断则添加的后缀（默认：'...'）
        
    Returns:
        截断后的文本
        
    Examples:
        >>> truncate_text("This is a long text", 10)
        'This is...'
    """
    if not text or len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def split_into_chunks(text: str, chunk_size: int, overlap: int = 0) -> List[str]:
    """
    将文本分割为指定大小的块，可选择重叠。
    
    Args:
        text: 要分割的输入文本
        chunk_size: 每个块的最大大小
        overlap: 块之间重叠的字符数
        
    Returns:
        文本块列表
        
    Examples:
        >>> split_into_chunks("abcdefghij", 4, 2)
        ['abcd', 'cdef', 'efgh', 'ghij']
    """
    if not text:
        return []
    
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")
    
    chunks = []
    start = 0
    step = chunk_size - overlap
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += step
    
    return chunks


def extract_sentences(text: str) -> List[str]:
    """
    使用基于标点符号的简单分割从文本中提取句子。
    
    Args:
        text: 输入文本
        
    Returns:
        句子列表
        
    Examples:
        >>> extract_sentences("Hello world. How are you?")
        ['Hello world', 'How are you']
    """
    if not text:
        return []
    
    # 按句子结束标点符号分割
    sentences = re.split(r'[.!?]+', text)
    
    # 清理并过滤空句子
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


def normalize_text(text: str, 
                  lowercase: bool = False,
                  remove_punct: bool = False,
                  remove_digits: bool = False) -> str:
    """
    使用多个选项规范化文本。
    
    Args:
        text: 输入文本
        lowercase: 转换为小写
        remove_punct: 删除标点符号
        remove_digits: 删除数字
        
    Returns:
        规范化后的文本
        
    Examples:
        >>> normalize_text("Test123!", lowercase=True, remove_punct=True)
        'test123'
    """
    if not text:
        return text
    
    result = text
    
    if lowercase:
        result = result.lower()
    
    if remove_punct:
        result = re.sub(r'[^\w\s]', '', result)
    
    if remove_digits:
        result = re.sub(r'\d', '', result)
    
    return result.strip()


def count_words(text: str) -> int:
    """
    统计文本中的单词数。
    
    Args:
        text: 输入文本
        
    Returns:
        单词数量
        
    Examples:
        >>> count_words("Hello world")
        2
    """
    if not text:
        return 0
    
    words = text.split()
    return len(words)


def count_characters(text: str, include_spaces: bool = True) -> int:
    """
    统计文本中的字符数。
    
    Args:
        text: 输入文本
        include_spaces: 是否统计空格
        
    Returns:
        字符数量
        
    Examples:
        >>> count_characters("Hello world")
        11
        >>> count_characters("Hello world", include_spaces=False)
        10
    """
    if not text:
        return 0
    
    if include_spaces:
        return len(text)
    else:
        return len(text.replace(' ', ''))


def is_empty_or_whitespace(text: str) -> bool:
    """
    检查文本是否为空或仅包含空白字符。
    
    Args:
        text: 输入文本
        
    Returns:
        如果为空或仅包含空白字符则返回 True
        
    Examples:
        >>> is_empty_or_whitespace("")
        True
        >>> is_empty_or_whitespace("   ")
        True
        >>> is_empty_or_whitespace("test")
        False
    """
    if not text:
        return True
    
    return not text.strip()
