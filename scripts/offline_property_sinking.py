import logging
import sys
import os
import json
from neo4j import GraphDatabase

# Ensure project root is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llama.config import NEO4J_CONFIG

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("property_sinking.log")
    ]
)
logger = logging.getLogger(__name__)

class PropertySinker:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_CONFIG["url"],
            auth=(NEO4J_CONFIG["username"], NEO4J_CONFIG["password"])
        )
        self.max_length_threshold = 15  # 长度超过此值的节点将被视为候选属性
        
        # 允许下沉的关系类型
        self.target_relations = [
            "定义", "definition", 
            "描述", "description", 
            "特征", "feature", 
            "表现", "manifestation", 
            "表现为", # User mentioned "表现为"
            "含义", "meaning",
            "explanation", "解释"
        ]

    def close(self):
        self.driver.close()

    def run(self):
        logger.info("开始属性下沉任务...")
        
        # 1. 查找所有符合条件的“长文本节点”及其关系
        # 条件：节点名称长度 > threshold 且 作为关系的各种 Target
        
        query = """
        MATCH (s)-[r]->(t)
        WHERE size(t.name) > $threshold
        RETURN elementId(s) as source_id, s.name as source_name, 
               elementId(r) as rel_id, type(r) as rel_type, 
               elementId(t) as target_id, t.name as target_text
        """
        
        # Fallback for old Neo4j
        query_legacy = """
        MATCH (s)-[r]->(t)
        WHERE size(t.name) > $threshold
        RETURN id(s) as source_id, s.name as source_name, 
               id(r) as rel_id, type(r) as rel_type, 
               id(t) as target_id, t.name as target_text
        """
        
        processed_count = 0
        deleted_nodes = set()
        
        with self.driver.session() as session:
            try:
                result = session.run(query, threshold=self.max_length_threshold)
            except:
                result = session.run(query_legacy, threshold=self.max_length_threshold)
                
            records = list(result)
            logger.info(f"找到 {len(records)} 个潜在的下沉候选关系")
            
            for record in records:
                source_id = record["source_id"]
                source_name = record["source_name"]
                rel_type = record["rel_type"]
                target_id = record["target_id"]
                target_text = record["target_text"]
                
                # 检查关系类型是否在允许列表中
                # 或者如果文本非常长 (>30)，我们可能放宽关系限制，假设它是描述
                is_target_rel = rel_type in self.target_relations
                is_very_long = len(target_text) > 30
                
                if not (is_target_rel or is_very_long):
                    continue
                
                # 决定属性名称
                # 如果关系类型是 "定义"，属性名设为 "definition"
                # 否则默认为 "description" 或保留关系名
                prop_name = "description"
                if rel_type in ["定义", "definition", "含义", "解释"]:
                    prop_name = "definition"
                elif rel_type in ["表现", "表现为", "manifestation"]:
                    prop_name = "manifestation"
                elif rel_type in ["特征", "feature"]:
                    prop_name = "feature"
                
                logger.info(f"Sinking: ({source_name}) --[{rel_type}]--> ({target_text[:20]}...) => Property: {prop_name}")
                
                # 执行更新
                # 1. 设置属性
                # 注意：如果属性已存在，追加还是覆盖？这里选择追加 (append)
                update_query = f"""
                MATCH (s) WHERE elementId(s) = $source_id
                SET s.{prop_name} = 
                    CASE 
                        WHEN s.{prop_name} IS NULL THEN $text 
                        ELSE s.{prop_name} + '; ' + $text 
                    END
                """
                update_query_legacy = f"""
                MATCH (s) WHERE id(s) = $source_id
                SET s.{prop_name} = 
                    CASE 
                        WHEN s.{prop_name} IS NULL THEN $text 
                        ELSE s.{prop_name} + '; ' + $text 
                    END
                """
                
                try:
                    if isinstance(source_id, int):
                        session.run(update_query_legacy, source_id=source_id, text=target_text)
                    else:
                        session.run(update_query, source_id=source_id, text=target_text)
                except Exception as e:
                    logger.error(f"更新属性失败: {e}")
                    continue

                # 2. 标记删除目标节点
                # 只有当目标节点不再有其他重要连接时才删除？
                # 简单起见，我们先删除该关系。如果节点孤立了，再删除节点。
                
                # 删除关系
                del_rel_query = "MATCH ()-[r]->() WHERE elementId(r) = $rel_id DELETE r"
                del_rel_query_legacy = "MATCH ()-[r]->() WHERE id(r) = $rel_id DELETE r"
                
                try:
                    if isinstance(record["rel_id"], int):
                        session.run(del_rel_query_legacy, rel_id=record["rel_id"])
                    else:
                        session.run(del_rel_query, rel_id=record["rel_id"])
                except Exception as e:
                    logger.error(f"删除关系失败: {e}")
                
                deleted_nodes.add(target_id)
                processed_count += 1
        
        # 3. 清理孤立的长文本节点
        if deleted_nodes:
            logger.info(f"检查 {len(deleted_nodes)} 个节点是否孤立并清理...")
            cleaned_count = 0
            with self.driver.session() as session:
                for nid in deleted_nodes:
                    # 检查是否还有其他关系
                    check_query = "MATCH (n) WHERE elementId(n) = $id AND size((n)--()) > 0 RETURN count(n)"
                    check_query_legacy = "MATCH (n) WHERE id(n) = $id AND size((n)--()) > 0 RETURN count(n)"
                    
                    try:
                        if isinstance(nid, int):
                            res = session.run(check_query_legacy, id=nid)
                        else:
                            res = session.run(check_query, id=nid)
                            
                        if res.single()[0] == 0:
                            # 孤立，删除
                            del_node_query = "MATCH (n) WHERE elementId(n) = $id DELETE n"
                            del_node_query_legacy = "MATCH (n) WHERE id(n) = $id DELETE n"
                            
                            if isinstance(nid, int):
                                session.run(del_node_query_legacy, id=nid)
                            else:
                                session.run(del_node_query, id=nid)
                            cleaned_count += 1
                    except Exception as e:
                        pass
                        
            logger.info(f"清理了 {cleaned_count} 个孤立的长文本节点")
            
        logger.info(f"属性下沉完成，共处理 {processed_count} 条关系")

if __name__ == "__main__":
    sinker = PropertySinker()
    try:
        sinker.run()
    finally:
        sinker.close()
