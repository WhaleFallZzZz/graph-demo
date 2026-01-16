#!/usr/bin/env python3
"""
动态扩充标准实体库脚本

功能：
1. 统计图谱中出现频率 > 阈值 的非标准实体
2. 生成更新后的配置代码
3. 可选：自动更新 enhanced_entity_extractor.py 和 config.py

使用方法：
    python scripts/expand_standard_entities.py [--threshold 3] [--auto-update]
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import Counter
import argparse
import logging

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入配置和标准实体
try:
    from llama.config import NEO4J_CONFIG
    from llama.enhanced_entity_extractor import StandardTermMapper
except ImportError as e:
    print(f"导入失败: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def connect_neo4j():
    """连接到 Neo4j 数据库"""
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            NEO4J_CONFIG["url"],
            auth=(NEO4J_CONFIG["username"], NEO4J_CONFIG["password"])
        )
        # 测试连接
        with driver.session() as session:
            session.run("RETURN 1")
        logger.info("✅ Neo4j 连接成功")
        return driver
    except Exception as e:
        logger.error(f"❌ Neo4j 连接失败: {e}")
        return None


def get_entity_frequencies(driver, min_frequency: int = 3) -> Dict[str, int]:
    """
    统计实体在图谱中的出现频率
    
    Args:
        driver: Neo4j driver
        min_frequency: 最小频率阈值
    
    Returns:
        实体名称到频率的字典
    """
    logger.info(f"正在统计实体频率（阈值 >= {min_frequency}）...")
    
    with driver.session() as session:
        # 统计实体作为节点的出现次数（通过关系数量）
        # 一个实体可能作为头节点或尾节点出现在多个关系中
        query = """
        MATCH (e:__Entity__)
        OPTIONAL MATCH (e)-[r1]->()
        OPTIONAL MATCH ()-[r2]->(e)
        WITH e, COUNT(DISTINCT r1) + COUNT(DISTINCT r2) as frequency
        WHERE frequency >= $min_freq
        RETURN e.name as entity_name, frequency
        ORDER BY frequency DESC
        """
        
        result = session.run(query, min_freq=min_frequency)
        entity_freqs = {}
        for record in result:
            entity_name = record["entity_name"]
            frequency = record["frequency"]
            if entity_name:  # 确保名称不为空
                entity_freqs[entity_name] = frequency
        
        logger.info(f"找到 {len(entity_freqs)} 个频率 >= {min_frequency} 的实体")
        return entity_freqs


def filter_non_standard_entities(entity_freqs: Dict[str, int], standard_entities: Set[str]) -> Dict[str, int]:
    """
    过滤出非标准实体
    
    Args:
        entity_freqs: 实体频率字典
        standard_entities: 标准实体集合
    
    Returns:
        非标准实体的频率字典
    """
    non_standard = {}
    for entity, freq in entity_freqs.items():
        if entity not in standard_entities:
            non_standard[entity] = freq
    
    logger.info(f"过滤后，找到 {len(non_standard)} 个非标准实体")
    return non_standard


def classify_entities(entities: List[str], driver) -> Dict[str, List[str]]:
    """
    根据实体类型对实体进行分类
    
    Args:
        entities: 实体名称列表
        driver: Neo4j driver
    
    Returns:
        按类型分类的实体字典
    """
    classification = {
        "疾病": [],
        "症状体征": [],
        "部位": [],
        "检查参数": [],
        "治疗防控": [],
        "其他": []
    }
    
    with driver.session() as session:
        for entity in entities:
            # 查询实体的类型
            query = """
            MATCH (e:__Entity__ {name: $entity_name})
            RETURN COALESCE(e.type, e.label, '其他') as entity_type
            LIMIT 1
            """
            result = session.run(query, entity_name=entity)
            record = result.single()
            
            if record:
                entity_type = record["entity_type"]
                # 映射到分类
                if entity_type in classification:
                    classification[entity_type].append(entity)
                else:
                    classification["其他"].append(entity)
            else:
                classification["其他"].append(entity)
    
    return classification


def generate_updated_code(non_standard_entities: Dict[str, int], classification: Dict[str, List[str]], existing_entities: Set[str]) -> str:
    """
    生成更新后的代码（仅包含新增实体，需要手动合并到原代码）
    
    Args:
        non_standard_entities: 非标准实体频率字典
        classification: 按类型分类的实体
        existing_entities: 现有的标准实体集合（用于参考，不输出）
    
    Returns:
        更新后的代码字符串（仅新增实体部分）
    """
    code_lines = []
    code_lines.append("# 新增的高频非标准实体（频率 >= 阈值）")
    code_lines.append("# 请将这些实体添加到 enhanced_entity_extractor.py 的 STANDARD_ENTITIES 中")
    code_lines.append("# 格式：按类型分类，每行最多5个实体")
    code_lines.append("")
    
    # 按分类输出新增实体
    type_order = ["疾病", "症状体征", "部位", "检查参数", "治疗防控", "其他"]
    for entity_type in type_order:
        entities_in_type = classification.get(entity_type, [])
        if entities_in_type:
            code_lines.append(f"# {entity_type}({len(entities_in_type)} 个)")
            # 按频率排序
            entities_with_freq = [(e, non_standard_entities.get(e, 0)) for e in entities_in_type]
            entities_with_freq.sort(key=lambda x: x[1], reverse=True)
            
            # 每行最多5个实体
            entities_list = [f'"{e}"' for e, _ in entities_with_freq]
            for i in range(0, len(entities_list), 5):
                batch = entities_list[i:i+5]
                is_last_batch = (i + 5 >= len(entities_list))
                is_last_type = (entity_type == type_order[-1] or not any(classification.get(t) for t in type_order[type_order.index(entity_type)+1:]))
                comma = "" if (is_last_batch and is_last_type) else ","
                code_lines.append(f"    {', '.join(batch)}{comma}")
            code_lines.append("")  # 类型之间空一行
    
    return "\n".join(code_lines)


def update_enhanced_entity_extractor(new_entities: Set[str], file_path: Path):
    """
    更新 enhanced_entity_extractor.py 中的 STANDARD_ENTITIES
    
    Args:
        new_entities: 新的实体集合
        file_path: enhanced_entity_extractor.py 文件路径
    """
    logger.info(f"正在更新 {file_path}...")
    
    # 读取原文件
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找 STANDARD_ENTITIES 的定义
    # 这里需要根据实际文件结构来更新
    # 由于 STANDARD_ENTITIES 是一个 set，我们需要合并新旧实体
    
    # 简单实现：找到 STANDARD_ENTITIES = { 到 } 的部分并替换
    # 这需要更复杂的解析，暂时只生成建议代码
    logger.warning("自动更新功能需要手动实现，请查看生成的代码并手动更新文件")


def generate_report(non_standard_entities: Dict[str, int], classification: Dict[str, List[str]], threshold: int):
    """
    生成统计报告
    
    Args:
        non_standard_entities: 非标准实体频率字典
        classification: 按类型分类的实体
        threshold: 频率阈值
    """
    print("\n" + "="*80)
    print("标准实体库扩充报告")
    print("="*80)
    print(f"\n频率阈值: >= {threshold}")
    print(f"找到的非标准实体总数: {len(non_standard_entities)}")
    print(f"\n按类型分类:")
    
    type_order = ["疾病", "症状体征", "部位", "检查参数", "治疗防控", "其他"]
    for entity_type in type_order:
        entities_in_type = classification.get(entity_type, [])
        if entities_in_type:
            print(f"\n{entity_type} ({len(entities_in_type)} 个):")
            entities_with_freq = [(e, non_standard_entities.get(e, 0)) for e in entities_in_type]
            entities_with_freq.sort(key=lambda x: x[1], reverse=True)
            for entity, freq in entities_with_freq[:20]:  # 只显示前20个
                print(f"  - {entity}: {freq} 次")
            if len(entities_with_freq) > 20:
                print(f"  ... 还有 {len(entities_with_freq) - 20} 个实体")
    
    print("\n" + "="*80)
    print("频率最高的20个非标准实体:")
    sorted_entities = sorted(non_standard_entities.items(), key=lambda x: x[1], reverse=True)
    for i, (entity, freq) in enumerate(sorted_entities[:20], 1):
        print(f"{i:2d}. {entity:30s} ({freq:3d} 次)")


def main():
    parser = argparse.ArgumentParser(description="动态扩充标准实体库")
    parser.add_argument("--threshold", type=int, default=3, help="最小频率阈值（默认: 3）")
    parser.add_argument("--auto-update", action="store_true", help="自动更新配置文件（暂未实现）")
    parser.add_argument("--output", type=str, help="输出文件路径（可选）")
    
    args = parser.parse_args()
    
    # 连接 Neo4j
    driver = connect_neo4j()
    if not driver:
        logger.error("无法连接到 Neo4j，请检查配置")
        sys.exit(1)
    
    try:
        # 获取标准实体集合
        standard_entities = StandardTermMapper.STANDARD_ENTITIES
        logger.info(f"当前标准实体数量: {len(standard_entities)}")
        
        # 统计实体频率
        entity_freqs = get_entity_frequencies(driver, args.threshold)
        
        # 过滤非标准实体
        non_standard_entities = filter_non_standard_entities(entity_freqs, standard_entities)
        
        if not non_standard_entities:
            logger.info("没有找到符合条件的非标准实体")
            return
        
        # 分类实体
        logger.info("正在分类实体...")
        classification = classify_entities(list(non_standard_entities.keys()), driver)
        
        # 生成报告
        generate_report(non_standard_entities, classification, args.threshold)
        
        # 生成更新后的代码
        updated_code = generate_updated_code(non_standard_entities, classification, standard_entities)
        
        # 输出代码
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(updated_code)
            logger.info(f"代码已保存到: {output_path}")
        else:
            print("\n" + "="*80)
            print("更新后的 STANDARD_ENTITIES 代码:")
            print("="*80)
            print(updated_code)
            print("\n提示: 使用 --output 参数保存到文件")
        
        if args.auto_update:
            logger.warning("自动更新功能暂未实现，请手动更新 enhanced_entity_extractor.py")
        
    finally:
        driver.close()


if __name__ == "__main__":
    main()
