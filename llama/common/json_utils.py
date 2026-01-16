"""
JSON 解析和处理工具模块

提供健壮的 JSON 解析和处理功能，用于处理 LLM 输出和各种 JSON 格式。

主要功能：
- 带错误处理的安全 JSON 解析
- LLM 输出解析和标准化
- JSON 语法修复和更正
- 从混合文本中提取 JSON
- JSON 结构验证
- CSV 与 JSON 转换
"""

import json
import re
import logging
from typing import Any, List, Dict, Optional, Union

try:
    import json5
    HAS_JSON5 = True
except ImportError:
    HAS_JSON5 = False
    logger = logging.getLogger(__name__)
    logger.warning("json5 库未安装，将使用标准 json 库")

logger = logging.getLogger(__name__)


class JSONParseError(Exception):
    """
    JSON 解析错误
    
    自定义异常类，用于表示 JSON 解析过程中发生的错误。
    """
    pass


def safe_json_parse(text: str, default: Any = None) -> Any:
    """
    安全解析 JSON 文本，带错误处理和自动修复
    
    优先使用 json5 库进行解析，因为它支持更宽松的 JSON 格式。
    如果解析失败，会尝试修复语法错误后再解析。
    
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
    
    # 优先使用 json5 库进行解析
    if HAS_JSON5:
        try:
            result = json5.loads(text)
            logger.debug("JSON parsing succeeded with json5 library")
            return result
        except Exception as e:
            logger.debug(f"json5 parsing failed: {e}")
    
    # 首先尝试直接解析（使用标准 json 库）
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
    解析 LLM 输出，提取带有类型信息的实体
    
    处理各种 LLM 输出格式并尝试提取结构化实体数据。
    支持以下格式：
    - JSON 数组
    - JSON 对象（包含 entities 键）
    - 单个 JSON 对象
    - 逐行格式
    
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
    将实体列表标准化为标准格式
    
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
    将单个实体标准化为标准格式
    
    Args:
        entity: 实体字典（可能有各种键名）
        
    Returns:
        带有 'name' 和 'type' 键的标准化实体字典，或者完整的三元组格式
    """
    # 检查是否是三元组格式（包含head, relation, tail）
    if any(key in entity for key in ['head', 'relation', 'tail']):
        # 直接返回三元组格式的字典
        return {k: str(v) for k, v in entity.items()}
    
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


def _parse_line_by_line(text: str) -> List[Dict[str, str]]:
    """
    逐行解析文本，尝试提取实体
    
    Args:
        text: 要解析的文本
        
    Returns:
        实体字典列表
    """
    entities = []
    
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # 尝试解析为 JSON
        try:
            entity = json.loads(line)
            if isinstance(entity, dict):
                entities.append(_normalize_entity(entity))
        except:
            # 如果不是 JSON，作为简单文本处理
            entities.append({'name': line, 'type': 'unknown'})
    
    return entities


def fix_json_syntax(json_str: str) -> str:
    """
    尝试修复常见的 JSON 语法错误
    
    处理以下问题：
    - 键周围缺少引号
    - 尾随逗号
    - 单引号而不是双引号
    - 缺少右括号
    - 中文键名缺少引号
    - 缺少冒号
    - 不完整的 JSON 对象
    - 混合使用单引号和双引号
    
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
    
    # 移除注释（如果有）
    fixed = re.sub(r'//.*?\n', '\n', fixed)
    fixed = re.sub(r'/\*.*?\*/', '', fixed, flags=re.DOTALL)
    
    # 移除控制字符
    fixed = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', fixed)
    
    # 先统一所有单引号为双引号（这是最关键的一步）
    # 这样可以处理混合使用单引号和双引号的情况
    fixed = re.sub(r"'", '"', fixed)
    
    # 为未加引号的键添加引号（支持中文字符）
    # 匹配模式：{ 或 , 后面跟空白，然后是键名（可能包含中文），然后是冒号
    fixed = re.sub(
        r'([{,]\s*)([a-zA-Z_\u4e00-\u9fff][a-zA-Z0-9_\u4e00-\u9fff]*)\s*:',
        r'\1"\2":',
        fixed
    )
    
    # 移除尾随逗号（在 } 或 ] 之前）
    fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
    
    # 修复缺少右括号的情况
    # 统计括号数量并添加缺失的括号
    open_braces = fixed.count('{')
    close_braces = fixed.count('}')
    if open_braces > close_braces:
        fixed += '}' * (open_braces - close_braces)
    
    open_brackets = fixed.count('[')
    close_brackets = fixed.count(']')
    if open_brackets > close_brackets:
        fixed += ']' * (open_brackets - close_brackets)
    
    return fixed


def extract_json_from_text(text: str) -> Optional[str]:
    """
    从混合文本中提取 JSON 对象或数组
    
    处理包含嵌入在 markdown 代码块中或被其他文本包围的 JSON 的文本。
    改进版本：支持嵌套 JSON 结构的准确提取。
    
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
        # 尝试解析代码块中的JSON，选择第一个有效的
        for block in code_blocks:
            try:
                json.loads(block)
                return block
            except:
                continue
    
    # 改进的JSON提取：使用栈来匹配括号，支持嵌套结构
    def extract_json_with_stack(text: str, start_char: str, end_char: str) -> Optional[str]:
        """使用栈匹配括号来提取JSON"""
        start_indices = []
        for i, char in enumerate(text):
            if char == start_char:
                start_indices.append(i)
            elif char == end_char:
                if start_indices:
                    start = start_indices.pop()
                    if not start_indices:  # 匹配到最外层
                        json_candidate = text[start:i+1]
                        # 验证是否是有效的JSON
                        try:
                            json.loads(json_candidate)
                            return json_candidate
                        except:
                            continue
        return None
    
    # 先尝试提取JSON对象
    json_obj = extract_json_with_stack(text, '{', '}')
    if json_obj:
        return json_obj
    
    # 再尝试提取JSON数组
    json_array = extract_json_with_stack(text, '[', ']')
    if json_array:
        return json_array
    
    # 回退到简单正则匹配（对于简单情况）
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if json_match:
        candidate = json_match.group(0)
        try:
            json.loads(candidate)
            return candidate
        except:
            pass
    
    array_match = re.search(r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]', text, re.DOTALL)
    if array_match:
        candidate = array_match.group(0)
        try:
            json.loads(candidate)
            return candidate
        except:
            pass
    
    return None


def validate_json_structure(data: Any, 
                          required_keys: Optional[List[str]] = None,
                          expected_type: Optional[type] = None) -> bool:
    """
    根据要求验证 JSON 结构
    
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


def json_to_csv(data: List[Dict[str, Any]]) -> str:
    """
    将 JSON 对象列表转换为 CSV 格式
    
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
    将 CSV 格式转换为 JSON 对象列表
    
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
