import os
import sys
import logging
import numpy as np
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llama.config import NEO4J_CONFIG, setup_logging
from llama.factories import ModelFactory
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
        self.threshold = 0.90  # 降低阈值以合并更多相似实体
        self.review_threshold = 0.85 # 记录待复盘的阈值
        self.max_workers = max_workers  # 多线程数
        
        # 定义同义词映射表 (Normalized Name -> List of Synonyms) or (Synonym -> Normalized Name)
        # 这里使用 Synonym -> Normalized Name 方便查找
        self.synonym_map = {
            "OK镜": "角膜塑形镜",
            "ok镜": "角膜塑形镜",
            "Ortho-K": "角膜塑形镜",
            "全飞秒": "全飞秒激光手术",
            "全飞秒手术": "全飞秒激光手术",
            "SMILE": "全飞秒激光手术",
            "SMILE手术": "全飞秒激光手术",
            "smile": "全飞秒激光手术",
            "ICL": "ICL植入术",
            "icl": "ICL植入术",
            "AL": "眼轴长度",
            "IOP": "眼压",
            "K值": "角膜曲率",
        }

    def normalize_name(self, name: str) -> str:
        """
        对实体名称进行标准化处理：
        1. 单位统一 (如 +3.5D -> 350度)
        2. 同义词归一 (如 OK镜 -> 角膜塑形镜)
        """
        if not name:
            return name
            
        #1. 同义词映射 (优先完全匹配)
        if name in self.synonym_map:
            return self.synonym_map[name]
            
        #2. 单位标准化
        # 匹配屈光度格式：+3.50D, -6.00D, 3.5D, +3.50DS
        diopter_pattern = r'([+-]?\d+\.?\d*)\s*[Dd][Ss]?'
        match = re.fullmatch(diopter_pattern, name.strip())
        if match:
            try:
                val = float(match.group(1))
                # 统一转换为"度"：数值 * 100
                # 保留正负号逻辑：通常近视用负号，远视用正号。
                # 但如果用户输入 "350度"，通常指绝对值。
                # 考虑到 "+3.5D" -> "350度" 的需求，这里直接转换数值。
                # 如果是负数，保留负号，如 "-3.0D" -> "-300度"
                degrees = int(val * 100)
                return f"{degrees}度"
            except:
                pass
                
        #3. 简单清洗
        # 比如去除多余空格
        return name.strip()

    def close(self):
        self.driver.close()
        
    def get_all_entities(self) -> List[Dict]:
        """获取所有实体节点（仅 __Entity__ 节点）"""
        logger.info("正在从 Neo4j 获取所有实体节点...")
        # 只查询 __Entity__ 标签的节点，避免匹配到 Chunk 或 Document 节点
        # 使用 id(n) 以兼容旧版本，如果是 Neo4j 5+ 可以用 elementId(n)
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
                logger.warning(f"使用 elementId() 失败，尝试使用 id(): {e}")
                query_fallback = """
                MATCH (n:__Entity__)
                WHERE n.name IS NOT NULL
                RETURN id(n) as id, n.name as name
                """
                result = session.run(query_fallback)
                entities = [{"id": record["id"], "name": record["name"]} for record in result]
                
        logger.info(f"共获取到 {len(entities)} 个实体节点")
        return entities

    def _compute_batch_embeddings(self, batch: List[str]) -> List[List[float]]:
        """计算单个批次的嵌入"""
        try:
            return self.embed_model.get_text_embedding_batch(batch)
        except Exception as e:
            logger.error(f"批次嵌入计算失败: {e}")
            # 返回零向量
            dim = 1024
            return [[0.0] * dim] * len(batch)

    def compute_embeddings(self, entities: List[Dict]) -> List[List[float]]:
        """多线程计算或获取嵌入"""
        names = [e["name"] for e in entities]
        logger.info(f"正在计算 {len(names)} 个实体的嵌入（使用 {self.max_workers} 个线程）...")
        
        batch_size = 100
        batches = [names[i:i+batch_size] for i in range(0, len(names), batch_size)]
        
        embeddings = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._compute_batch_embeddings, batch) for batch in batches]
            
            for i, future in enumerate(as_completed(futures)):
                try:
                    batch_embeddings = future.result()
                    embeddings.extend(batch_embeddings)
                    logger.info(f"进度: {min((i+1)*batch_size, len(names))}/{len(names)}")
                except Exception as e:
                    logger.error(f"批次处理失败: {e}")
                    # 填充零向量
                    dim = 1024
                    embeddings.extend([[0.0] * dim] * batch_size)
        
        return embeddings

    def align_and_merge(self):
        entities = self.get_all_entities()
        if not entities:
            logger.warning("未找到任何实体节点")
            return
        
        # 1. 预处理：计算规范化名称
        logger.info("正在进行名称规范化处理...")
        for e in entities:
            e["original_name"] = e["name"]
            e["normalized_name"] = self.normalize_name(e["name"])
            if e["normalized_name"] != e["original_name"]:
                logger.debug(f"Normalize: '{e['original_name']}' -> '{e['normalized_name']}'")

        # 2. 计算 Embeddings (使用规范化名称)
        # 注意：这里我们提取 normalized_name 进行计算，这样同义词的 Embedding 将完全一致
        norm_names = [e["normalized_name"] for e in entities]
        
        logger.info(f"正在计算 {len(norm_names)} 个实体的嵌入(基于规范化名称)...")
        # 复用 compute_embeddings 逻辑，但这里手动调用以支持 list 输入
        batch_size = 100
        batches = [norm_names[i:i+batch_size] for i in range(0, len(norm_names), batch_size)]
        
        embeddings = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._compute_batch_embeddings, batch) for batch in batches]
            
            for i, future in enumerate(as_completed(futures)):
                try:
                    batch_embeddings = future.result()
                    embeddings.extend(batch_embeddings)
                    logger.info(f"进度: {min((i+1)*batch_size, len(norm_names))}/{len(norm_names)}")
                except Exception as e:
                    logger.error(f"批次 {i} 嵌入计算失败: {e}")
                    dim = 1024 
                    embeddings.extend([[0.0] * dim] * batch_size)
        
        logger.info("计算相似度矩阵...")
        # 转换为 numpy 数组
        emb_matrix = np.array(embeddings)
        
        # 归一化
        norm = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        # 避免除零
        norm[norm == 0] = 1e-10
        normalized_emb = emb_matrix / norm
        
        # 计算相似度矩阵 (N x N)
        similarity_matrix = np.dot(normalized_emb, normalized_emb.T)
        
        # 使用并查集 (Union-Find) 进行聚类
        parent = list(range(len(entities)))
        def find(i):
            if parent[i] != i:
                parent[i] = find(parent[i])
            return parent[i]
        
        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_j] = root_i
        
        logger.info(f"查找相似度 > {self.review_threshold} 的节点对...")
        log_entries = []
        
        count = len(entities)
        merge_candidates_count = 0
        review_candidates_count = 0
        
        # 遍历上三角矩阵
        for i in range(count):
            for j in range(i + 1, count):
                # 1. 优先合并：规范化名称完全相同 (包括同义词和单位转换)
                if entities[i]["normalized_name"] == entities[j]["normalized_name"]:
                    union(i, j)
                    continue
                    
                # 2. 相似度合并
                sim = similarity_matrix[i][j]
                
                # 记录高相似度对
                if sim >= self.review_threshold:
                    status = "MERGE" if sim >= self.threshold else "REVIEW"
                    log_entries.append(f"{status}: '{entities[i]['original_name']}' <-> '{entities[j]['original_name']}' (Score: {sim:.4f})")
                    
                    if sim >= self.threshold:
                        union(i, j)
                        merge_candidates_count += 1
                    else:
                        review_candidates_count += 1
        
        # 写入日志
        log_file = "offline_alignment.log"
        with open(log_file, "w") as f:
            f.write(f"Alignment Report\n")
            f.write(f"Merge Threshold: {self.threshold}\n")
            f.write(f"Review Threshold: {self.review_threshold}\n")
            f.write("================================================\n")
            # 按分数降序排列日志
            log_entries.sort(key=lambda x: float(x.split("Score: ")[1].strip(")")), reverse=True)
            f.write("\n".join(log_entries))
            
        logger.info(f"已发现 {merge_candidates_count} 对合并项，{review_candidates_count} 对待复盘项，日志已写入 {log_file}")
        
        # 分组
        groups = {}
        for i in range(count):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)
        
        # 过滤掉单元素组
        multi_entity_groups = {k: v for k, v in groups.items() if len(v) > 1}
        
        logger.info(f"识别出 {len(multi_entity_groups)} 个需要合并的组")
        
        # 执行合并
        merge_success = 0
        merge_fail = 0
        
        for group_id, indices in multi_entity_groups.items():
            # 找到主节点（第一个）
            primary_idx = indices[0]
            primary = entities[primary_idx]
            primary_id = primary["id"]
            target_name = primary["normalized_name"]
            
            # 其他节点
            other_indices = indices[1:]
            other_ids = [entities[idx]["id"] for idx in other_indices]
            
            if not other_ids:
                continue
            
            logger.info(f"合并组: 主节点 '{primary['original_name']}' <- {[entities[idx]['original_name'] for idx in other_indices]}")
            
            # 执行 Neo4j 合并操作
            with self.driver.session() as session:
                try:
                    # 使用 elementId() 或 id() 根据版本
                    if isinstance(primary_id, int):
                        # 处理 outgoing 关系：other -> target（只处理 Entity 节点之间的关系）
                        query_outgoing = """
                         MATCH (other:__Entity__)
                        WHERE id(other) IN $other_ids
                        MATCH (other)-[r]->(target:__Entity__)
                        WHERE target IS NOT NULL AND r IS NOT NULL AND id(target) <> $primary_id
                        WITH id(target) as target_id, type(r) as rel_type, properties(r) as rel_props
                        WITH collect(DISTINCT {target_id: target_id, rel_type: rel_type, rel_props: rel_props}) as rels
                        UNWIND rels as rel
                        MATCH (primary:__Entity__) WHERE id(primary) = $primary_id
                        MATCH (target:__Entity__) WHERE id(target) = rel.target_id
                        CALL apoc.create.relationship(primary, rel.rel_type, target, CASE WHEN rel.rel_props IS NULL THEN {} ELSE rel.rel_props END)
                        YIELD rel as created_rel
                        RETURN count(created_rel) as count
                        """
                        
                        # 处理 incoming 关系：source -> other（只处理 Entity 节点之间的关系）
                        query_incoming = """
                        MATCH (other:__Entity__)
                        WHERE id(other) IN $other_ids
                        MATCH (source:__Entity__)-[r]->(other)
                        WHERE source IS NOT NULL AND r IS NOT NULL AND id(source) <> $primary_id
                        WITH id(source) as source_id, type(r) as rel_type, properties(r) as rel_props
                        WITH collect(DISTINCT {source_id: source_id, rel_type: rel_type, rel_props: rel_props}) as rels
                        UNWIND rels as rel
                        MATCH (primary:__Entity__) WHERE id(primary) = $primary_id
                        MATCH (source:__Entity__) WHERE id(source) = rel.source_id
                        CALL apoc.create.relationship(source, rel.rel_type, primary, CASE WHEN rel.rel_props IS NULL THEN {} ELSE rel.rel_props END)
                        YIELD rel as created_rel
                        RETURN count(created_rel) as count
                        """
                        
                        # 删除 other 节点
                        query_delete = """
                        MATCH (other:__Entity__)
                        WHERE id(other) IN $other_ids
                        DETACH DELETE other
                        """
                        
                        # 执行三个操作
                        session.run(query_outgoing, primary_id=primary_id, other_ids=other_ids)
                        session.run(query_incoming, primary_id=primary_id, other_ids=other_ids)
                        session.run(query_delete, primary_id=primary_id, other_ids=other_ids)
                    else:
                        # 处理 outgoing 关系：other -> target（只处理 Entity 节点之间的关系）
                        query_outgoing = """
                        MATCH (other:__Entity__)
                        WHERE elementId(other) IN $other_ids
                        MATCH (other)-[r]->(target:__Entity__)
                        WHERE target IS NOT NULL AND r IS NOT NULL AND elementId(target) <> $primary_id
                        WITH elementId(target) as target_id, type(r) as rel_type, properties(r) as rel_props
                        WITH collect(DISTINCT {target_id: target_id, rel_type: rel_type, rel_props: rel_props}) as rels
                        UNWIND rels as rel
                        MATCH (primary:__Entity__) WHERE elementId(primary) = $primary_id
                        MATCH (target:__Entity__) WHERE elementId(target) = rel.target_id
                        CALL apoc.create.relationship(primary, rel.rel_type, target, CASE WHEN rel.rel_props IS NULL THEN {} ELSE rel.rel_props END)
                        YIELD rel as created_rel
                        RETURN count(created_rel) as count
                        """
                        
                        # 处理 incoming 关系：source -> other（只处理 Entity 节点之间的关系）
                        query_incoming = """
                             MATCH (other:__Entity__)
                        WHERE elementId(other) IN $other_ids
                        MATCH (source:__Entity__)-[r]->(other)
                        WHERE source IS NOT NULL AND r IS NOT NULL AND elementId(source) <> $primary_id
                        WITH elementId(source) as source_id, type(r) as rel_type, properties(r) as rel_props
                        WITH collect(DISTINCT {source_id: source_id, rel_type: rel_type, rel_props: rel_props}) as rels
                        UNWIND rels as rel
                        MATCH (primary:__Entity__) WHERE elementId(primary) = $primary_id
                        MATCH (source:__Entity__) WHERE elementId(source) = rel.source_id
                        CALL apoc.create.relationship(source, rel.rel_type, primary, CASE WHEN rel.rel_props IS NULL THEN {} ELSE rel.rel_props END)
                        YIELD rel as created_rel
                        RETURN count(created_rel) as count
                        """
                        
                        # 删除 other 节点
                        query_delete = """
                        MATCH (other:__Entity__)
                        WHERE elementId(other) IN $other_ids
                        DETACH DELETE other
                        """
                        
                        # 执行三个操作
                        session.run(query_outgoing, primary_id=primary_id, other_ids=other_ids)
                        session.run(query_incoming, primary_id=primary_id, other_ids=other_ids)
                        session.run(query_delete, primary_id=primary_id, other_ids=other_ids)
                    merge_success += 1
                except Exception as e:
                    logger.error(f"合并失败 '{primary['original_name']}': {e}")
                    merge_fail += 1
                        
        logger.info(f"合并/更新完成: 成功 {merge_success} 组, 失败 {merge_fail} 组")

if __name__ == "__main__":
    # 可以通过环境变量或命令行参数设置线程数
    max_workers = int(os.getenv('ALIGNMENT_WORKERS', '4'))
    aligner = OfflineEntityAligner(max_workers=max_workers)
    try:
        aligner.align_and_merge()
    except Exception as e:
        logger.error(f"脚本执行出错: {e}")
    finally:
        aligner.close()
