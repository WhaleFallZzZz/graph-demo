"""
增强的JSON解析工具模块
专门处理LLM输出中的各种JSON格式问题
"""

import json
import re
import logging
from typing import List, Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)

def safe_json_parse(text: str) -> Optional[List[Dict[str, Any]]]:
    """
    安全解析JSON，尝试多种修复策略
    
    Args:
        text: 可能包含JSON的文本
        
    Returns:
        解析成功的JSON列表，失败返回None
    """
    if not text or not text.strip():
        return None
    
    # 策略1: 直接尝试解析
    try:
        result = json.loads(text.strip())
        if isinstance(result, list):
            return result
        elif isinstance(result, dict):
            return [result]
    except (json.JSONDecodeError, ValueError):
        pass
    
    # 策略2: 提取可能的JSON部分
    json_text = extract_json_text(text)
    if json_text:
        try:
            result = json.loads(json_text)
            if isinstance(result, list):
                return result
            elif isinstance(result, dict):
                return [result]
        except (json.JSONDecodeError, ValueError):
            pass
    
    # 策略3: 修复常见的JSON语法错误
    fixed_text = fix_json_syntax(text)
    if fixed_text:
        try:
            result = json.loads(fixed_text)
            if isinstance(result, list):
                return result
            elif isinstance(result, dict):
                return [result]
        except (json.JSONDecodeError, ValueError):
            pass
    
    return None

def extract_json_text(text: str) -> Optional[str]:
    """
    从文本中提取最可能的JSON部分
    
    Args:
        text: 包含JSON的文本
        
    Returns:
        提取的JSON字符串，失败返回None
    """
    # 移除注释和多余空白
    text = re.sub(r'//.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    
    # Strip markdown code blocks
    text = re.sub(r'^```json\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^```\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'```$', '', text, flags=re.MULTILINE)
    
    text = text.strip()
    
    # 尝试找到JSON数组（最完整的形式）
    patterns = [
        # 完整的JSON数组
        (r'\[\s*\{[\s\S]*?\}\s*\]', True),
        # 多个JSON对象组成的数组
        (r'\[\s*\{.*?\},\s*\{[\s\S]*?\}\s*\]', True),
        # 单个JSON对象
        (r'\{\s*"head"[\s\S]*?\}', False),
        # 多个JSON对象
        (r'\{[\s\S]*?\}', False),
    ]
    
    for pattern, is_array in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            # 优先选择最长的匹配
            best_match = max(matches, key=len)
            if is_array:
                return best_match
            else:
                # 如果是单个对象，包装成数组
                return f"[{best_match}]"
    
    return None

def fix_json_syntax(text: str) -> Optional[str]:
    """
    修复常见的JSON语法错误
    
    Args:
        text: 可能有语法错误的JSON文本
        
    Returns:
        修复后的JSON字符串，失败返回None
    """
    # 提取JSON部分
    json_part = extract_json_text(text)
    if not json_part:
        return None
    
    # 修复策略
    fixes = [
        # 修复尾随逗号
        (r',\s*}', '}'),
        (r',\s*]', ']'),
        # 修复单引号
        (r"'", '"'),
        # 修复中文引号 - 使用Unicode编码避免正则表达式问题
        (r'\u201c|\u201d|\u2018|\u2019', '"'),
        # 修复多余的逗号
        (r',\s*([}\]])', r'\1'),
        # 修复未闭合的字符串
        (r'"([^"]*)$', r'"\1"'),
    ]
    
    fixed = json_part
    for pattern, replacement in fixes:
        fixed = re.sub(pattern, replacement, fixed)
    
    # 验证修复结果
    try:
        result = json.loads(fixed)
        return fixed
    except json.JSONDecodeError as e:
        logger.debug(f"修复后的JSON仍然无效: {e}")
        return None

def extract_triplets_from_json(json_data: List[Dict[str, Any]]) -> List[Tuple[str, str, str]]:
    """
    从JSON数据中提取三元组
    
    Args:
        json_data: 包含三元组信息的JSON列表
        
    Returns:
        (head, relation, tail)元组列表
    """
    triplets = []
    
    if not json_data:
        logger.warning("JSON数据为空")
        return triplets
    
    for i, item in enumerate(json_data):
        if not isinstance(item, dict):
            logger.warning(f"条目 {i} 不是字典，跳过")
            continue
        
        # 安全提取字段，处理None值
        head = str(item.get("head", "") or "").strip()
        head_type = str(item.get("head_type", "") or "").strip()
        relation = str(item.get("relation", "") or "").strip()
        tail = str(item.get("tail", "") or "").strip()
        tail_type = str(item.get("tail_type", "") or "").strip()
        
        # 验证字段完整性
        if all([head, head_type, relation, tail, tail_type]):
            triplets.append((head, relation, tail))
            logger.debug(f"成功提取三元组: {head} - {relation} - {tail}")
        else:
            logger.warning(f"条目 {i} 字段不完整，跳过: {item}")
    
    return triplets

def extract_triplets_with_regex(text: str) -> List[Tuple[str, str, str]]:
    """
    使用正则表达式提取三元组（备用方法）
    
    Args:
        text: LLM输出文本
        
    Returns:
        (head, relation, tail)元组列表
    """
    triplets = []
    
    # 多种正则表达式模式
    patterns = [
        # 标准JSON格式
        r'"head"\s*:\s*"([^"]*)".*?"relation"\s*:\s*"([^"]*)".*?"tail"\s*:\s*"([^"]*)"',
        # 简化格式
        r'"head"\s*:\s*"([^"]*)".*"relation"\s*:\s*"([^"]*)".*"tail"\s*:\s*"([^"]*)"',
        # 更宽松的模式 - 避免使用字符集引起正则表达式错误
        r'head["\s:]+["\']([^"\']+)["\'].*?relation["\s:]+["\']([^"\']+)["\'].*?tail["\s:]+["\']([^"\']+)["\']',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        for head, relation, tail in matches:
            if all([h.strip() for h in [head, relation, tail]]):
                triplets.append((head.strip(), relation.strip(), tail.strip()))
    
    logger.info(f"正则表达式解析到 {len(triplets)} 个三元组")
    return triplets

def extract_triplets_with_types_regex(text: str) -> List[Dict[str, str]]:
    """
    使用正则表达式提取带类型的三元组（增强备用方法）
    
    Args:
        text: LLM输出文本
        
    Returns:
        包含完整字段的字典列表
    """
    results = []
    
    # 尝试捕获所有字段的宽容模式
    # 这一模式尝试匹配 standard JSON 结构，但也允许一定的灵活性
    pattern = r'"head"\s*:\s*"(.*?)"\s*,\s*"head_type"\s*:\s*"(.*?)"\s*,\s*"relation"\s*:\s*"(.*?)"\s*,\s*"tail"\s*:\s*"(.*?)"\s*,\s*"tail_type"\s*:\s*"(.*?)"'
    
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    for head, head_type, relation, tail, tail_type in matches:
        if all([h.strip() for h in [head, relation, tail]]):
            results.append({
                "head": head.strip(),
                "head_type": head_type.strip() or "概念",
                "relation": relation.strip(),
                "tail": tail.strip(),
                "tail_type": tail_type.strip() or "概念"
            })
            
    # 如果上面的模式没匹配到，尝试另一种常见的字段顺序 (tail 在 tail_type 之前)
    if not results:
        pattern2 = r'"head"\s*:\s*"(.*?)"\s*,\s*"head_type"\s*:\s*"(.*?)"\s*,\s*"relation"\s*:\s*"(.*?)"\s*,\s*"tail_type"\s*:\s*"(.*?)"\s*,\s*"tail"\s*:\s*"(.*?)"'
        matches2 = re.findall(pattern2, text, re.DOTALL | re.IGNORECASE)
        for head, head_type, relation, tail_type, tail in matches2:
            if all([h.strip() for h in [head, relation, tail]]):
                results.append({
                    "head": head.strip(),
                    "head_type": head_type.strip() or "概念",
                    "relation": relation.strip(),
                    "tail": tail.strip(),
                    "tail_type": tail_type.strip() or "概念"
                })

    logger.info(f"带类型的正则表达式解析提取到 {len(results)} 个结果")
    return results

def parse_llm_output_with_types(llm_output: str) -> List[Dict[str, str]]:
    """
    解析LLM输出，尽可能保留类型信息
    优先尝试JSON解析，失败则使用增强的正则解析
    """
    if not llm_output or not llm_output.strip():
        return []
        
    # 添加调试日志
    logger.info(f"开始解析LLM输出，长度: {len(llm_output)}")
    logger.info(f"LLM输出预览: {llm_output[:200]}...")
    
    # 1. 尝试JSON解析
    logger.info("尝试JSON解析...")
    json_data = safe_json_parse(llm_output)
    logger.info(f"JSON解析结果: {json_data is not None}")
    if json_data:
        logger.info(f"JSON解析成功，得到 {len(json_data)} 个条目")
        results = []
        for item in json_data:
            if isinstance(item, dict) and item.get("head") and item.get("relation") and item.get("tail"):
                results.append({
                    "head": str(item.get("head", "")).strip(),
                    "head_type": str(item.get("head_type", "")).strip() or "概念",
                    "relation": str(item.get("relation", "")).strip(),
                    "tail": str(item.get("tail", "")).strip(),
                    "tail_type": str(item.get("tail_type", "")).strip() or "概念"
                })
        if results:
            logger.info(f"从JSON数据中提取到 {len(results)} 个有效三元组")
            return results
        else:
            logger.warning("JSON解析成功但没有找到有效的三元组数据")
    else:
        logger.info("JSON解析失败，将使用正则表达式")

    # 2. JSON解析失败，使用增强正则
    regex_results = extract_triplets_with_types_regex(llm_output)
    logger.info(f"正则表达式解析得到 {len(regex_results)} 个结果")
    return regex_results

def parse_llm_output_to_triplets(llm_output: str) -> List[Tuple[str, str, str]]:
    """
    主解析函数：从LLM输出中提取三元组
    
    Args:
        llm_output: LLM的原始输出
        
    Returns:
        (head, relation, tail)元组列表
    """
    if not llm_output or not llm_output.strip():
        logger.warning("LLM输出为空")
        return []
    
    logger.info(f"开始解析LLM输出，长度: {len(llm_output)}")
    
    # 使用新的带类型解析函数，然后降级为元组
    dicts = parse_llm_output_with_types(llm_output)
    triplets = []
    for d in dicts:
        triplets.append((d["head"], d["relation"], d["tail"]))
        
    if triplets:
        logger.info(f"解析提取 {len(triplets)} 个三元组")
        return triplets
        
    # 如果新的解析失败（极少数情况），尝试旧的宽松正则作为最后防线
    logger.info(f"高级解析失败，尝试基础正则表达式解析...")
    return extract_triplets_with_regex(llm_output)

# 向后兼容的函数名
parse_dynamic_triplets = parse_llm_output_to_triplets