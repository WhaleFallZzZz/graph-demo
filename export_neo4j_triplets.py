#!/usr/bin/env python3
"""
导出 Neo4j 中所有三元组到 JSON 文件
"""

import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime, date
import neo4j.time

# 添加项目根目录和llama目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "llama"))

# 导入配置和工厂
from config import setup_logging
from factories import GraphStoreFactory

# 设置日志
logger = setup_logging()

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        # Handle Neo4j specific types
        if isinstance(obj, (neo4j.time.DateTime, neo4j.time.Date, neo4j.time.Time)):
             return obj.isoformat()
        try:
            return str(obj)
        except:
            return super().default(obj)

def export_triplets(output_file="neo4j_export.json"):
    """导出所有三元组"""
    logger.info("开始连接图数据库...")
    
    # 创建图存储实例
    graph_store = GraphStoreFactory.create_graph_store()
    
    if not graph_store:
        logger.error("无法创建图存储连接")
        return False
        
    store_type = type(graph_store).__name__
    logger.info(f"已连接图存储: {store_type}")
    
    if "Neo4jPropertyGraphStore" not in store_type:
        logger.warning("未检测到 Neo4j 连接，当前使用内存存储，可能无数据可导出")
    
    logger.info("正在查询所有三元组...")
    
    # 查询所有关系的 Cypher 语句
    # 排除系统内部使用的节点和关系 (Embedding, __Embedding__, __Vector__)
    cypher_query = """
        MATCH (s)-[r]->(t)
        WHERE NOT s:Embedding AND NOT s:__Embedding__ AND NOT s:__Vector__
          AND NOT t:Embedding AND NOT t:__Embedding__ AND NOT t:__Vector__
          AND NOT type(r) CONTAINS 'EMBEDDING' 
          AND NOT type(r) CONTAINS 'embedding'
        RETURN 
            elementId(s) as source_id, 
            labels(s) as source_labels, 
            properties(s) as source_props,
            elementId(t) as target_id, 
            labels(t) as target_labels, 
            properties(t) as target_props,
            type(r) as rel_type, 
            properties(r) as rel_props,
            elementId(r) as rel_id
    """
    
    try:
        results = graph_store.structured_query(cypher_query)
        logger.info(f"查询成功，获取到 {len(results)} 条关系")
        
        triplets = []
        
        for record in results:
            # 处理源节点
            source_labels = record.get("source_labels", [])
            source_label = source_labels[0] if source_labels else "Unknown"
            source_props = record.get("source_props", {})
            
            # 处理目标节点
            target_labels = record.get("target_labels", [])
            target_label = target_labels[0] if target_labels else "Unknown"
            target_props = record.get("target_props", {})
            
            # 处理关系
            rel_type = record.get("rel_type", "UNKNOWN")
            rel_props = record.get("rel_props", {})
            
            triplet = {
                "source": {
                    "id": record.get("source_id"),
                    "label": source_label,
                    "properties": source_props
                },
                "target": {
                    "id": record.get("target_id"),
                    "label": target_label,
                    "properties": target_props
                },
                "relationship": {
                    "id": record.get("rel_id"),
                    "type": rel_type,
                    "properties": rel_props
                }
            }
            triplets.append(triplet)
            
        # 导出到 JSON
        output_path = Path(output_file).absolute()
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(triplets, f, ensure_ascii=False, indent=2, cls=DateTimeEncoder)
            
        logger.info(f"✅ 成功导出 {len(triplets)} 个三元组到文件: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"导出失败: {e}")
        return False

if __name__ == "__main__":
    export_triplets()
