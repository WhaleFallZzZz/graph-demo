#!/usr/bin/env python3
"""
导出 Neo4j 中的节点和边，用于评判实体召回率
只保留核心信息，去除无用的元数据
"""

import sys
import os
import json
import logging
from pathlib import Path
from collections import defaultdict

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

# 导入配置和工厂
from llama.config import setup_logging
from llama.factories import GraphStoreFactory

# 设置日志
logger = setup_logging()

def export_nodes_and_edges(output_file="neo4j_entities.json"):
    """导出节点和边，用于评判实体召回率"""
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
    
    logger.info("正在查询所有节点和边...")
    
    # 查询所有节点（排除系统节点）
    nodes_query = """
        MATCH (n)
        WHERE NOT n:Embedding AND NOT n:__Embedding__ AND NOT n:__Vector__
          AND n.name IS NOT NULL
        RETURN DISTINCT 
            n.name as name,
            n.type as type,
            labels(n) as labels
        ORDER BY n.name
    """
    
    # 查询所有边（排除系统边）
    edges_query = """
        MATCH (s)-[r]->(t)
        WHERE NOT s:Embedding AND NOT s:__Embedding__ AND NOT s:__Vector__
          AND NOT t:Embedding AND NOT t:__Embedding__ AND NOT t:__Vector__
          AND s.name IS NOT NULL AND t.name IS NOT NULL
          AND NOT type(r) CONTAINS 'EMBEDDING' 
          AND NOT type(r) CONTAINS 'embedding'
        RETURN DISTINCT 
            s.name as source,
            t.name as target,
            type(r) as relation_type
        ORDER BY s.name, t.name, type(r)
    """
    
    try:
        # 查询节点
        logger.info("正在查询节点...")
        nodes_results = graph_store.structured_query(nodes_query)
        logger.info(f"查询成功，获取到 {len(nodes_results)} 个节点")
        
        # 查询边
        logger.info("正在查询边...")
        edges_results = graph_store.structured_query(edges_query)
        logger.info(f"查询成功，获取到 {len(edges_results)} 条边")
        
        # 处理节点数据
        nodes = []
        node_set = set()
        for record in nodes_results:
            name = record.get("name", "").strip()
            if not name:
                continue
            
            # 去重
            if name in node_set:
                continue
            node_set.add(name)
            
            node_type = record.get("type", "")
            labels = record.get("labels", [])
            
            # 如果没有 type，尝试从 labels 中提取
            if not node_type and labels:
                # 过滤掉 __Node__ 等系统标签
                type_labels = [l for l in labels if l not in ["__Node__", "Entity"]]
                node_type = type_labels[0] if type_labels else ""
            
            nodes.append({
                "name": name,
                "type": node_type if node_type else "Entity"
            })
        
        # 处理边数据
        edges = []
        edge_set = set()
        for record in edges_results:
            source = record.get("source", "").strip()
            target = record.get("target", "").strip()
            relation_type = record.get("relation_type", "").strip()
            
            if not source or not target or not relation_type:
                continue
            
            # 去重
            edge_key = f"{source}|{relation_type}|{target}"
            if edge_key in edge_set:
                continue
            edge_set.add(edge_key)
            
            edges.append({
                "source": source,
                "target": target,
                "relation_type": relation_type
            })
        
        # 构建输出数据
        output_data = {
            "summary": {
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "export_time": None
            },
            "nodes": nodes,
            "edges": edges
        }
        
        # 导出到 JSON
        output_path = Path(output_file).absolute()
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ 成功导出 {len(nodes)} 个节点和 {len(edges)} 条边到文件: {output_path}")
        
        # 打印统计信息
        logger.info("\n=== 导出统计 ===")
        logger.info(f"节点总数: {len(nodes)}")
        logger.info(f"边总数: {len(edges)}")
        
        # 统计节点类型分布
        type_distribution = defaultdict(int)
        for node in nodes:
            type_distribution[node["type"]] += 1
        logger.info("\n节点类型分布:")
        for node_type, count in sorted(type_distribution.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {node_type}: {count}")
        
        # 统计关系类型分布
        relation_distribution = defaultdict(int)
        for edge in edges:
            relation_distribution[edge["relation_type"]] += 1
        logger.info("\n关系类型分布:")
        for relation_type, count in sorted(relation_distribution.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {relation_type}: {count}")
        
        return True
        
    except Exception as e:
        logger.error(f"导出失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    export_nodes_and_edges()
