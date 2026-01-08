"""
LLM 输出的 JSON 解析和验证工具。

本模块提供健壮的 JSON 解析函数，包括：
- 带错误处理的安全 JSON 解析
- 带类型保留的 LLM 输出解析
- JSON 语法修复和更正
- 从混合文本中提取 JSON
- 结构验证
"""

import json
import re
import logging
from typing import Any, List, Dict, Optional, Union

logger = logging.getLogger(__name__)


class JSONParseError(Exception):
    """Custom exception for JSON parsing errors."""
    pass


def safe_json_parse(text: str, default: Any = None) -> Any:
    """
    安全解析 JSON 文本，带错误处理和自动修复。
    
    Args:
        text: 要解析的 JSON 字符串
        default: 如果解析失败则返回的默认值
        
    Returns:
        解析后的 JSON 对象或默认值
        
    Examples:
        >>> safe_json_parse('{"key": "value"}')
        {'key': 'value'}
        >>> safe_json_parse('invalid', default={})
        {}
    """
    if not text or not text.strip():
        logger.debug("Empty or whitespace-only JSON input")
        return default if default is not None else None
    
    # 首先尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.debug(f"Initial JSON parsing failed: {e}")
    
    # 如果直接解析失败，尝试修复语法
    fixed_text = fix_json_syntax(text)
    
    try:
        result = json.loads(fixed_text)
        logger.info("JSON parsing succeeded after syntax fix")
        return result
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parsing failed even after syntax fix: {e}")
        
        # 尝试从文本中提取JSON
        extracted = extract_json_from_text(text)
        if extracted:
            logger.debug("Attempting to parse extracted JSON from text")
            try:
                result = json.loads(extracted)
                logger.info("JSON parsing succeeded from extracted text")
                return result
            except json.JSONDecodeError:
                logger.warning("Failed to parse extracted JSON")
        
        return default if default is not None else None
    except Exception as e:
        logger.error(f"Unexpected error parsing JSON: {e}")
        return default if default is not None else None


def parse_llm_output(llm_output: str) -> List[Dict[str, str]]:
    """
    解析 LLM 输出，提取带有类型信息的实体。
    
    处理各种 LLM 输出格式并尝试提取结构化实体数据。
    
    Args:
        llm_output: 原始 LLM 输出文本
        
    Returns:
        包含实体数据的字典列表
        
    Examples:
        >>> parse_llm_output('[{"name": "test", "type": "disease"}]')
        [{'name': 'test', 'type': 'disease'}]
    """
    if not llm_output or not llm_output.strip():
        return []
    
    # 首先尝试直接 JSON 解析
    json_data = safe_json_parse(llm_output)
    
    if json_data is not None:
        # 处理不同的输出格式
        if isinstance(json_data, list):
            return _normalize_entity_list(json_data)
        elif isinstance(json_data, dict):
            # 检查是否有 entities 键
            if 'entities' in json_data:
                return _normalize_entity_list(json_data['entities'])
            # 否则作为单个实体处理
            return [_normalize_entity(json_data)]
    
    # 尝试从文本中提取 JSON
    extracted = extract_json_from_text(llm_output)
    if extracted:
        return parse_llm_output(extracted)
    
    # 尝试逐行解析
    return _parse_line_by_line(llm_output)


def _normalize_entity_list(entities: List[Any]) -> List[Dict[str, str]]:
    """
    将实体列表标准化为标准格式。
    
    Args:
        entities: 实体对象列表（各种格式）
        
    Returns:
        标准化的实体字典列表
    """
    normalized = []
    
    for entity in entities:
        if isinstance(entity, dict):
            normalized.append(_normalize_entity(entity))
        elif isinstance(entity, str):
            normalized.append({'name': entity, 'type': 'unknown'})
        else:
            logger.warning(f"Skipping invalid entity type: {type(entity)}")
    
    return normalized


def _normalize_entity(entity: Dict[str, Any]) -> Dict[str, str]:
    """
    将单个实体标准化为标准格式。
    
    Args:
        entity: 实体字典（可能有各种键名）
        
    Returns:
        带有 'name' 和 'type' 键的标准化实体字典
    """
    normalized = {'name': '', 'type': 'unknown'}
    
    # 处理各种键名变体
    name_keys = ['name', 'entity', 'text', 'value', 'label']
    type_keys = ['type', 'category', 'class', 'entity_type']
    
    for key in name_keys:
        if key in entity:
            normalized['name'] = str(entity[key])
            break
    
    for key in type_keys:
        if key in entity:
            normalized['type'] = str(entity[key])
            break
    
    # 复制任何其他字段
    for key, value in entity.items():
        if key not in name_keys and key not in type_keys:
            normalized[key] = str(value)
    
    return normalized


def fix_json_syntax(json_str: str) -> str:
    """
    尝试修复常见的 JSON 语法错误。
    
    处理以下问题：
    - 键周围缺少引号
    - 尾随逗号
    - 单引号而不是双引号
    - 缺少右括号
    
    Args:
        json_str: 可能格式错误的 JSON 字符串
        
    Returns:
        修复后的 JSON 字符串
        
    Examples:
        >>> fix_json_syntax("{'key': 'value',}")
        '{"key": "value"}'
    """
    if not json_str:
        return json_str
    
    fixed = json_str
    
    # 将单引号替换为双引号
    fixed = re.sub(r"'([^']*)'", r'"\1"', fixed)
    
    # 移除尾随逗号
    fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
    
    # 为未加引号的键添加引号
    fixed = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', fixed)
    
    # 移除注释（如果有）
    fixed = re.sub(r'//.*?\n', '\n', fixed)
    fixed = re.sub(r'/\*.*?\*/', '', fixed, flags=re.DOTALL)
    
    # 移除控制字符
    fixed = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', fixed)
    
    return fixed


def extract_json_from_text(text: str) -> Optional[str]:
    """
    从混合文本中提取 JSON 对象或数组。
    
    处理包含嵌入在 markdown 代码块中或被其他文本包围的 JSON 的文本。
    
    Args:
        text: 包含 JSON 的混合文本
        
    Returns:
        提取的 JSON 字符串，如果未找到则返回 None
        
    Examples:
        >>> extract_json_from_text('Here is the data: {"key": "value"}')
        '{"key": "value"}'
        >>> extract_json_from_text('```json\\n{"key": "value"}\\n```')
        '{"key": "value"}'
    """
    if not text:
        return None
    
    # 尝试在 markdown 代码块中查找 JSON
    code_blocks = re.findall(r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```', text, re.DOTALL)
    if code_blocks:
        return code_blocks[0]
    
    # 尝试查找 JSON 对象
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if json_match:
        return json_match.group(0)
    
    # 尝试查找 JSON 数组
    array_match = re.search(r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]', text, re.DOTALL)
    if array_match:
        return array_match.group(0)
    
    return None


def validate_json_structure(data: Any, 
                          required_keys: Optional[List[str]] = None,
                          expected_type: Optional[type] = None) -> bool:
    """
    根据要求验证 JSON 结构。
    
    Args:
        data: 要验证的已解析 JSON 数据
        required_keys: 必需键列表（用于字典对象）
        expected_type: 预期的数据类型
        
    Returns:
        如果有效则返回 True，否则返回 False
        
    Examples:
        >>> validate_json_structure({'name': 'test'}, required_keys=['name'])
        True
        >>> validate_json_structure([], expected_type=list)
        True
    """
    if data is None:
        return False
    
    # 检查类型
    if expected_type is not None and not isinstance(data, expected_type):
        logger.warning(f"Type mismatch: expected {expected_type}, got {type(data)}")
        return False
    
    # 检查字典的必需键
    if isinstance(data, dict) and required_keys:
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            logger.warning(f"Missing required keys: {missing_keys}")
            return False
    
    return True


def parse_entity_triplets(text: str) -> List[Dict[str, str]]:
    """
    从 LLM 输出中解析实体三元组。
    
    三元组格式为：(头实体, 关系, 尾实体)
    
    Args:
        text: 包含三元组的 LLM 输出
        
    Returns:
        三元组字典列表
        
    Examples:
        >>> parse_entity_triplets("(disease, causes, symptom)")
        [{'head': 'disease', 'relation': 'causes', 'tail': 'symptom'}]
    """
    if not text:
        return []
    
    triplets = []
    
    # 匹配三元组的模式
    pattern = r'\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)]+)\s*\)'
    matches = re.findall(pattern, text)
    
    for match in matches:
        triplets.append({
            'head': match[0].strip(),
            'relation': match[1].strip(),
            'tail': match[2].strip()
        })
    
    return triplets


def format_json_output(data: Any, indent: int = 2) -> str:
    """
    将数据格式化为带有适当缩进的 JSON 字符串。
    
    Args:
        data: 要格式化的数据
        indent: 缩进空格数
        
    Returns:
        格式化的 JSON 字符串
        
    Examples:
        >>> format_json_output({'key': 'value'})
        '{\\n  "key": "value"\\n}'
    """
    try:
        return json.dumps(data, indent=indent, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to format JSON: {e}")
        return str(data)


def merge_json_objects(*objects: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并多个 JSON 对象，后面的对象会覆盖前面的对象。
    
    Args:
        *objects: 要合并的字典对象
        
    Returns:
        合并后的字典
        
    Examples:
        >>> merge_json_objects({'a': 1}, {'b': 2}, {'a': 3})
        {'a': 3, 'b': 2}
    """
    result = {}
    
    for obj in objects:
        if isinstance(obj, dict):
            result.update(obj)
        else:
            logger.warning(f"Skipping non-dict object in merge: {type(obj)}")
    
    return result


def flatten_json(data: Dict[str, Any], 
                 separator: str = '_',
                 parent_key: str = '') -> Dict[str, Any]:
    """
    展平嵌套的 JSON 结构。
    
    Args:
        data: 要展平的嵌套字典
        separator: 嵌套键的分隔符（默认：'_'）
        parent_key: 当前父键（内部使用）
        
    Returns:
        展平后的字典
        
    Examples:
        >>> flatten_json({'a': {'b': {'c': 1}}})
        {'a_b_c': 1}
    """
    items = []
    
    for key, value in data.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key
        
        if isinstance(value, dict):
            items.extend(flatten_json(value, separator, new_key).items())
        else:
            items.append((new_key, value))
    
    return dict(items)


def unflatten_json(data: Dict[str, Any], separator: str = '_') -> Dict[str, Any]:
    """
    将之前展平的 JSON 结构还原为嵌套结构。
    
    Args:
        data: 展平后的字典
        separator: 展平键中使用的分隔符
        
    Returns:
        嵌套字典
        
    Examples:
        >>> unflatten_json({'a_b_c': 1})
        {'a': {'b': {'c': 1}}}
    """
    result = {}
    
    for key, value in data.items():
        parts = key.split(separator)
        current = result
        
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        current[parts[-1]] = value
    
    return result


def json_to_csv(data: List[Dict[str, Any]]) -> str:
    """
    将 JSON 对象列表转换为 CSV 格式。
    
    Args:
        data: 具有一致键的字典列表
        
    Returns:
        CSV 格式字符串
        
    Examples:
        >>> json_to_csv([{'a': 1, 'b': 2}, {'a': 3, 'b': 4}])
        'a,b\\n1,2\\n3,4'
    """
    if not data:
        return ''
    
    # 获取所有唯一键
    keys = set()
    for item in data:
        if isinstance(item, dict):
            keys.update(item.keys())
    
    keys = sorted(keys)
    
    # 构建 CSV
    lines = [','.join(keys)]
    
    for item in data:
        if isinstance(item, dict):
            values = [str(item.get(key, '')) for key in keys]
            lines.append(','.join(values))
    
    return '\n'.join(lines)


def csv_to_json(csv_text: str) -> List[Dict[str, str]]:
    """
    将 CSV 格式转换为 JSON 对象列表。
    
    Args:
        csv_text: CSV 格式字符串
        
    Returns:
        字典列表
        
    Examples:
        >>> csv_to_json('a,b\\n1,2\\n3,4')
        [{'a': '1', 'b': '2'}, {'a': '3', 'b': '4'}]
    """
    if not csv_text:
        return []
    
    lines = csv_text.strip().split('\n')
    
    if not lines:
        return []
    
    # 解析表头
    keys = lines[0].split(',')
    
    # 解析数据行
    result = []
    for line in lines[1:]:
        values = line.split(',')
        if len(values) == len(keys):
            row = dict(zip(keys, values))
            result.append(row)
    
    return result
