import os
import sys
import logging
import numpy as np
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llama.config import NEO4J_CONFIG, setup_logging, RATE_LIMIT_CONFIG
from llama.factories import ModelFactory
from llama.neo4j_text_sanitizer import Neo4jTextSanitizer
from neo4j import GraphDatabase

logger = setup_logging()

import re

class OfflineEntityAligner:
    def __init__(self, max_workers: int = 4):
        self.driver = GraphDatabase.driver(
            NEO4J_CONFIG["url"],
            auth=(NEO4J_CONFIG["username"], NEO4J_CONFIG["password"])
        )
        self.embed_model = ModelFactory.create_embedding_model()
        self.threshold = 0.92  # 稍微提高阈值以增加准确性
        self.review_threshold = 0.85 # 记录待复盘的阈值
        self.max_workers = max_workers
        
        # 同义词映射表
        self.synonym_map = {
            "OK镜": "角膜塑形镜",
            "ok镜": "角膜塑形镜",
            "Ortho-K": "角膜塑形镜",
            "全飞秒": "全飞秒激光手术",
            "全飞秒手术": "全飞秒激光手术",
            "SMILE": "全飞秒激光手术",
            "ICL": "ICL植入术",
            "AL": "眼轴长度",
            "IOP": "眼压",
            "K值": "角膜曲率",
        }

    def normalize_name(self, name: str) -> str:
        """对实体名称进行标准化处理"""
        if not name:
            return name
            
        # 1. 使用官方清理工具
        name = Neo4jTextSanitizer.sanitize_node_name(name)
            
        # 2. 同义词映射
        if name in self.synonym_map:
            return self.synonym_map[name]
            
        # 3. 单位标准化 (+3.5D -> 350度)
        diopter_pattern = r'([+-]?\d+\.?\d*)\s*[Dd][Ss]?'
        match = re.fullmatch(diopter_pattern, name.strip())
        if match:
            try:
                val = float(match.group(1))
                degrees = int(val * 100)
                return f"{degrees}度"
            except:
                pass
                
        return name.strip()

    def close(self):
        self.driver.close()
        
    def get_all_entities(self) -> List[Dict]:
        """获取所有实体节点"""
        logger.info("正在从 Neo4j 获取所有实体节点...")
        # 统一使用 elementId (Neo4j 5+)
        query = """
        MATCH (n:__Entity__)
        WHERE n.name IS NOT NULL
        RETURN elementId(n) as id, n.name as name
        """
        
        entities = []
        with self.driver.session() as session:
            try:
                result = session.run(query)
                entities = [{"id": record["id"], "name": record["name"]} for record in result]
            except Exception as e:
                logger.error(f"获取实体失败: {e}")
                
        logger.info(f"共获取到 {len(entities)} 个实体节点")
        return entities

    def _compute_batch_embeddings(self, batch: List[str]) -> List[List[float]]:
        """计算批次嵌入"""
        try:
            return self.embed_model.get_text_embedding_batch(batch)
        except Exception as e:
            logger.error(f"嵌入计算失败: {e}")
            dim = 1024
            return [[0.0] * dim] * len(batch)

    def align_and_merge(self):
        entities = self.get_all_entities()
        if not entities:
            return
        
        # 1. 预处理与名称规约
        for e in entities:
            e["original_name"] = e["name"]
            e["normalized_name"] = self.normalize_name(e["name"])

        # 2. 计算 Embeddings (基于规范化名称)
        norm_names = [e["normalized_name"] for e in entities]
        batch_size = 50 # 减小批次大小以配合频控
        batches = [norm_names[i:i+batch_size] for i in range(0, len(norm_names), batch_size)]
        
        embeddings = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._compute_batch_embeddings, batch) for batch in batches]
            for i, future in enumerate(as_completed(futures)):
                embeddings.extend(future.result())
                if (i+1) % 10 == 0:
                    logger.info(f"Embedding进度: {min((i+1)*batch_size, len(norm_names))}/{len(norm_names)}")
        
        # 3. 计算相似度并聚类 (Union-Find)
        emb_matrix = np.array(embeddings)
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        normalized_emb = emb_matrix / norms
        similarity_matrix = np.dot(normalized_emb, normalized_emb.T)
        
        parent = list(range(len(entities)))
        def find(i):
            if parent[i] != i: parent[i] = find(parent[i])
            return parent[i]
        
        def union(i, j):
            root_i, root_j = find(i), find(j)
            if root_i != root_j: parent[root_j] = root_i

        log_entries = []
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                if entities[i]["normalized_name"] == entities[j]["normalized_name"]:
                    union(i, j)
                    continue
                
                sim = similarity_matrix[i][j]
                if sim >= self.review_threshold:
                    status = "MERGE" if sim >= self.threshold else "REVIEW"
                    log_entries.append(f"{status}: '{entities[i]['original_name']}' <-> '{entities[j]['original_name']}' ({sim:.4f})")
                    if sim >= self.threshold:
                        union(i, j)
        
        # 导出结果到文件
        import json
        report = {
            "total_entities": len(entities),
            "log": log_entries
        }
        with open("offline_alignment.log", "w") as f:
            f.write("Alignment Report\n" + "="*40 + "\n")
            f.write("\n".join(sorted(log_entries, reverse=True)))
            
        # 4. 执行 Neo4j 合并逻辑
        groups = {}
        for i in range(len(entities)):
            root = find(i)
            groups.setdefault(root, []).append(i)
        
        multi_groups = {k: v for k, v in groups.items() if len(v) > 1}
        logger.info(f"开始合并 {len(multi_groups)} 个实体组...")
        
        merge_success = 0
        for root_idx, indices in multi_groups.items():
            primary = entities[indices[0]]
            other_ids = [entities[idx]["id"] for idx in indices[1:]]
            
            with self.driver.session() as session:
                try:
                    # 合并关系并删除旧节点 (Cypher 优化)
                    # 1. 迁移 outgoing 关系
                    session.run("""
                        MATCH (other:__Entity__) WHERE elementId(other) IN $other_ids
                        MATCH (other)-[r]->(target:__Entity__) WHERE elementId(target) <> $primary_id
                        WITH r, type(r) as rel_type, properties(r) as rel_props, target
                        MATCH (primary:__Entity__) WHERE elementId(primary) = $primary_id
                        CALL apoc.create.relationship(primary, rel_type, rel_props, target) YIELD rel
                        RETURN count(rel)
                    """, primary_id=primary["id"], other_ids=other_ids)
                    
                    # 2. 迁移 incoming 关系
                    session.run("""
                        MATCH (other:__Entity__) WHERE elementId(other) IN $other_ids
                        MATCH (source:__Entity__)-[r]->(other) WHERE elementId(source) <> $primary_id
                        WITH r, type(r) as rel_type, properties(r) as rel_props, source
                        MATCH (primary:__Entity__) WHERE elementId(primary) = $primary_id
                        CALL apoc.create.relationship(source, rel_type, rel_props, primary) YIELD rel
                        RETURN count(rel)
                    """, primary_id=primary["id"], other_ids=other_ids)
                    
                    # 3. 删除旧节点
                    session.run("""
                        MATCH (n:__Entity__) WHERE elementId(n) IN $other_ids
                        DETACH DELETE n
                    """, other_ids=other_ids)
                    
                    merge_success += 1
                except Exception as e:
                    logger.error(f"合并 {primary['name']} 失败: {e}")
                    
        logger.info(f"清理完成: 成功合并 {merge_success} 组")

if __name__ == "__main__":
    max_workers = int(os.getenv('ALIGNMENT_WORKERS', '4'))
    aligner = OfflineEntityAligner(max_workers=max_workers)
    try:
        aligner.align_and_merge()
    except Exception as e:
        logger.error(f"脚本执行失败: {e}")
    finally:
        aligner.close()
