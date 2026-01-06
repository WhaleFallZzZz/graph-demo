
import logging
import asyncio
import sys
from pathlib import Path
import networkx as nx
from dotenv import load_dotenv

# 加载环境变量
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

import os
print(f"DEBUG: SILICONFLOW_LLM_MODEL={os.getenv('SILICONFLOW_LLM_MODEL')}")
print(f"DEBUG: Env path={env_path}, Exists={env_path.exists()}")

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent))

from llama.kg_manager import KnowledgeGraphManager
from llama.entity_resolution import EntityResolver
from llama_index.core import Document
from llama_index.core.graph_stores import SimplePropertyGraphStore

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_networkx_graph(triplets):
    """将三元组转换为NetworkX图，方便路径搜索"""
    G = nx.DiGraph()
    for triplet in triplets:
        head, relation, tail = triplet
        # relation 是 Relation 对象，包含 label
        G.add_edge(head.name, tail.name, relation=relation.label)
    return G

async def verify_logic(kg_manager, graph_store):
    """
    异步执行实体对齐和逻辑推理验证
    """
    # 4. 实体对齐 (Entity Resolution)
    logger.info("\n=== 阶段一：实体对齐 (Entity Resolution) ===")
    resolver = EntityResolver(kg_manager.embed_model)
    
    # 获取所有实体
    all_triplets = graph_store.get_triplets()
    entities = set()
    for t in all_triplets:
        entities.add(t[0].name)
        entities.add(t[2].name)
        
    logger.info(f"原始实体数量: {len(entities)}")
    logger.info(f"实体列表: {list(entities)}")
    
    # 查找重复
    duplicates = await resolver.find_duplicates(list(entities), threshold=0.85) # 稍微降低阈值以匹配"近视"和"青少年近视"
    
    if duplicates:
        logger.info(f"发现 {len(duplicates)} 对相似实体:")
        for e1, e2, score in duplicates:
            logger.info(f"  - '{e1}' <-> '{e2}' (相似度: {score:.4f})")
            
        # 应用合并
        merge_map = resolver.apply_resolution_to_triplets(all_triplets, duplicates)
        logger.info(f"生成合并映射: {merge_map}")
    else:
        logger.info("未发现需要合并的实体。")
        merge_map = {}

    # 5. 逻辑传导验证 (Multi-hop Reasoning)
    logger.info("\n=== 阶段二：逻辑传导验证 (Multi-hop Reasoning) ===")
    
    # 将图转换为 NetworkX 进行路径分析
    G = nx.DiGraph()
    for triplet in all_triplets:
        head = triplet[0].name
        relation = triplet[1].label
        tail = triplet[2].name
        
        # 应用实体对齐映射
        head = merge_map.get(head, head)
        tail = merge_map.get(tail, tail)
        
        G.add_edge(head, tail, relation=relation)
        
    logger.info(f"构建用于推理的图结构: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    
    # 定义查询路径
    start_node = "户外活动"
    end_node = "近视" 
    
    # 应用实体对齐映射到查询节点
    start_node = merge_map.get(start_node, start_node)
    end_node = merge_map.get(end_node, end_node)
    
    logger.info(f"尝试寻找路径: {start_node} -> ... -> {end_node}")
    
    # # Debug: Check Dopamine
    # dopamine = "多巴胺"
    # if dopamine in G:
    #     logger.info(f"节点 '{dopamine}' 存在于图中。")
    #     logger.info(f"入边: {list(G.in_edges(dopamine))}")
    #     logger.info(f"出边: {list(G.out_edges(dopamine))}")
    # else:
    #     logger.warning(f"节点 '{dopamine}' 不在图中！")
    
    try:
        if start_node not in G:
            logger.warning(f"起点 '{start_node}' 不在图中。现有节点: {list(G.nodes)}")
            return
        if end_node not in G:
            logger.warning(f"终点 '{end_node}' 不在图中。")
            return

        # 寻找所有简单路径 (限制长度)
        paths = list(nx.all_simple_paths(G, source=start_node, target=end_node, cutoff=5))
        
        if paths:
            logger.info(f"✅ 验证成功！找到 {len(paths)} 条逻辑传导路径：")
            for i, path in enumerate(paths):
                path_str = ""
                for j in range(len(path) - 1):
                    u = path[j]
                    v = path[j+1]
                    edge_data = G.get_edge_data(u, v)
                    relation = edge_data['relation']
                    path_str += f"[{u}] --({relation})--> "
                path_str += f"[{path[-1]}]"
                logger.info(f"路径 {i+1}: {path_str}")
                
            # dopamine_paths = [p for p in paths if "多巴胺" in p]
            # if dopamine_paths:
            #     logger.info("✨ 成功验证包含'多巴胺'的核心病理机制链条！")
            # else:
            #     logger.warning("⚠️ 未找到经过'多巴胺'的路径，请检查提取完整性。")
                
        else:
            logger.error("❌ 未找到逻辑通路。图谱可能存在断裂。")
            
    except nx.NetworkXNoPath:
        logger.error(f"❌ {start_node} 到 {end_node} 之间没有路径。")
    except Exception as e:
        logger.error(f"路径分析出错: {e}")

def main():
    # 1. 初始化 KG Manager
    kg_manager = KnowledgeGraphManager()
    if not kg_manager.initialize():
        logger.error("KG Manager 初始化失败")
        return

    # 2. 加载测试数据
    data_path = Path("data/logic_test.txt")
    if not data_path.exists():
        logger.error(f"数据文件不存在: {data_path}")
        return
        
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    documents = [Document(text=text)]
    logger.info(f"加载测试文档，长度: {len(text)}")

    # 3. 构建知识图谱
    logger.info("开始构建知识图谱...")
    
    from llama.factories import ExtractorFactory
    
    llm = kg_manager.llm
    if not llm:
        logger.error("LLM 未初始化")
        return

    # 创建提取器
    extractor = ExtractorFactory.create_extractor(llm)
    if not extractor:
        logger.error("提取器创建失败")
        return
        
    # 存入 GraphStore
    graph_store = kg_manager.graph_store
    
    # PropertyGraphIndex.from_documents 会调用提取逻辑 (内部调用 asyncio.run)
    from llama_index.core import PropertyGraphIndex
    
    index = PropertyGraphIndex.from_documents(
        documents,
        kg_extractors=[extractor],
        embed_model=kg_manager.embed_model,
        property_graph_store=graph_store,
        show_progress=True
    )
    logger.info("图谱构建完成。")
    
    # 执行异步验证逻辑
    asyncio.run(verify_logic(kg_manager, graph_store))

if __name__ == "__main__":
    main()
