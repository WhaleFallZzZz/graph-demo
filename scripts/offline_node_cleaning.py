import logging
import re
import json
import sys
import os
from typing import List, Dict, Set

# Ensure project root is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neo4j import GraphDatabase
from llama.config import NEO4J_CONFIG, RERANK_CONFIG
from llama.custom_siliconflow_rerank import CustomSiliconFlowRerank
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core import QueryBundle

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("node_cleaning.log")
    ]
)
logger = logging.getLogger(__name__)

class NodeCleaner:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_CONFIG["url"],
            auth=(NEO4J_CONFIG["username"], NEO4J_CONFIG["password"])
        )
        self.reranker = self._init_reranker()
        
        # Filtering Rules
        self.max_length = 15  # User: 15-20 characters
        
        # Stop Patterns (Regular Expressions)
        self.stop_patterns = [
            r"^在.*下$",          # 在...下
            r".*的时候$",         # ...的时候
            r"^如果",             # 如果...
            r"^当.*时$",          # 当...时
            r"^为了",             # 为了...
            r"^使得",             # 使得...
            r"^随着",             # 随着...
            r"^通过",             # 通过...
            r"^由于",             # 由于...
            r".*的影像$",         # ...的影像 (often descriptive)
            r".*的样子$",         # ...的样子
            r"^通俗易懂$",        # Adjective example
            r"^客观中肯$",        # Adjective example
            r"^歪曲和模糊的影像$", # Specific example
            r"^简单来说$",
            r"^换句话说$",
        ]
        
    def _init_reranker(self):
        try:
            # Use same config as entity extractor
            return CustomSiliconFlowRerank(
                model=RERANK_CONFIG.get("model", "BAAI/bge-reranker-v2-m3"),
                api_key=RERANK_CONFIG["api_key"],
                top_n=10 
            )
        except Exception as e:
            logger.warning(f"Reranker initialization failed: {e}. Semantic filtering will be disabled.")
            return None

    def get_all_entities(self):
        """Fetch all nodes with a name property"""
        with self.driver.session() as session:
            # Query all nodes with a name property, ignoring specific labels
            result = session.run("MATCH (n) WHERE n.name IS NOT NULL RETURN elementId(n) as id, n.name as name")
            return [{"id": record["id"], "name": record["name"]} for record in result]

    def delete_nodes(self, nodes_to_delete: List[Dict]):
        if not nodes_to_delete:
            return
        
        node_ids = [n["id"] for n in nodes_to_delete]
        
        # Log detailed backup before deletion
        with open("deleted_nodes_backup.json", "w", encoding="utf-8") as f:
            json.dump(nodes_to_delete, f, ensure_ascii=False, indent=2)
        logger.info(f"Backup of deleted nodes saved to deleted_nodes_backup.json")
        
        with self.driver.session() as session:
            # Batch delete
            batch_size = 1000
            total_deleted = 0
            for i in range(0, len(node_ids), batch_size):
                batch = node_ids[i:i+batch_size]
                session.run("""
                    MATCH (n:Entity)
                    WHERE elementId(n) IN $ids
                    DETACH DELETE n
                """, ids=batch)
                total_deleted += len(batch)
                logger.info(f"Deleted batch of {len(batch)} nodes. Progress: {total_deleted}/{len(node_ids)}")

    def matches_stop_patterns(self, name: str) -> bool:
        for pattern in self.stop_patterns:
            if re.search(pattern, name):
                return True
        return False

    def semantic_filter(self, candidates: List[Dict]) -> List[Dict]:
        """
        Use Rerank to find non-medical terms among candidates.
        Returns list of nodes to delete.
        """
        if not self.reranker or not candidates:
            return []

        to_delete = []
        batch_size = 50 
        
        logger.info(f"Starting semantic check for {len(candidates)} nodes...")
        
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i+batch_size]
            nodes = [NodeWithScore(node=TextNode(text=item["name"]), score=0.0) for item in batch]
            
            try:
                self.reranker.top_n = len(nodes)
                # Query "医学实体" (Medical Entity) to score relevance
                query = QueryBundle(query_str="医学实体")
                ranked_nodes = self.reranker.postprocess_nodes(nodes, query)
                
                # Threshold for "Garbage"
                # If score is very low, it's likely not a medical entity.
                # Common adjectives/stopwords usually score very low (<0.01 or <0.1).
                # User's threshold for *extraction* was 0.15. 
                # Here we are deleting existing nodes. Let's be consistent.
                # If score < 0.15, it's considered "irrelevant" by our own extraction standard.
                threshold = 0.15
                
                for node in ranked_nodes:
                    score = node.score if node.score is not None else 0.0
                    if score < threshold:
                        name = node.node.get_content()
                        original = next((x for x in batch if x["name"] == name), None)
                        if original:
                            logger.info(f"Semantic Filter: '{name}' (Score: {score:.4f}) < {threshold} -> MARK DELETE")
                            # Add reason for log
                            original["reason"] = f"Low Semantic Score ({score:.4f})"
                            to_delete.append(original)
            except Exception as e:
                logger.error(f"Rerank failed for batch: {e}")
                
        return to_delete

    def run(self):
        logger.info("Starting Node Cleaning Process...")
        all_nodes = self.get_all_entities()
        logger.info(f"Total nodes found: {len(all_nodes)}")
        
        nodes_to_delete = []
        candidates_for_semantic_check = []
        
        # Phase 1: Rule-based Filtering
        for node in all_nodes:
            name = node["name"]
            if not name:
                continue
                
            reason = None
            
            # 1. Length Filter
            if len(name) > self.max_length:
                reason = f"Length > {self.max_length}"
            
            # 2. Stop Pattern Filter
            elif self.matches_stop_patterns(name):
                reason = "Matches Stop Pattern"
            
            if reason:
                node["reason"] = reason
                nodes_to_delete.append(node)
                logger.info(f"Rule Filter: '{name}' -> MARK DELETE ({reason})")
            else:
                candidates_for_semantic_check.append(node)
        
        logger.info(f"Rule-based filtering marked {len(nodes_to_delete)} nodes for deletion.")
        
        # Phase 2: Semantic Filtering (Adjectives/Non-nouns)
        if candidates_for_semantic_check and self.reranker:
            logger.info(f"Proceeding to semantic check for {len(candidates_for_semantic_check)} remaining nodes...")
            semantic_deletes = self.semantic_filter(candidates_for_semantic_check)
            
            # 暂时禁用语义过滤的自动删除，仅记录日志
            # 原因：Rerank 模型对某些医学专有名词（如 '角膜塑形镜'）打分过低，存在误删风险。
            # 建议先通过规则过滤处理明显的噪音。
            # nodes_to_delete.extend(semantic_deletes)
            
            logger.info(f"Semantic filtering found {len(semantic_deletes)} potential non-entity nodes (Skipped deletion for safety).")
            
            # 将语义过滤结果写入单独的建议文件
            if semantic_deletes:
                with open("semantic_filter_suggestions.json", "w", encoding="utf-8") as f:
                    json.dump(semantic_deletes, f, ensure_ascii=False, indent=2)
                logger.info("Semantic filter suggestions saved to semantic_filter_suggestions.json")
        
        # Phase 3: Execution
        unique_ids = set()
        final_list = []
        for n in nodes_to_delete:
            if n["id"] not in unique_ids:
                unique_ids.add(n["id"])
                final_list.append(n)
        
        if final_list:
            logger.info(f"Deleting {len(final_list)} unique nodes...")
            self.delete_nodes(final_list)
            logger.info("Node cleaning completed.")
        else:
            logger.info("No nodes to delete.")
            
        self.driver.close()

if __name__ == "__main__":
    cleaner = NodeCleaner()
    cleaner.run()
