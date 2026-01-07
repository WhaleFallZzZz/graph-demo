import os
import sys
from neo4j import GraphDatabase

# Mimic the config loading
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "J.pQF!zg33haagT")
NEO4J_URL = os.environ.get("NEO4J_URL", "bolt://101.43.120.88:7687")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")

SYSTEM_SUFFIXES = ["系统", "平台", "软件", "客户端", "服务端", "APP"]
SYSTEM_KEYWORDS = ["电子病历系统", "emr", "his", "lis", "pacs", "ris", "挂号系统", "收费系统", "就诊平台", "预约平台", "管理系统", "医院信息系统", "医疗信息系统"]

def _build_system_filter():
    suffix_checks = " OR ".join([f"n.name ENDS WITH '{s}'" for s in SYSTEM_SUFFIXES])
    kw_checks = " OR ".join([f"toLower(n.name) CONTAINS '{k}'" for k in SYSTEM_KEYWORDS])
    return f"({suffix_checks}) OR ({kw_checks})"

def list_invalid_system_nodes(limit: int = 25):
    print("\n🔎 查找系统/平台/软件类无效节点...")
    try:
        driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        with driver.session(database=NEO4J_DATABASE) as session:
            where_clause = _build_system_filter()
            query = f"""
                MATCH (n)
                WHERE exists(n.name) AND ({where_clause})
                RETURN n, labels(n) as labels, elementId(n) as id
                LIMIT $limit
            """
            result = session.run(query, limit=limit)
            records = list(result)
            if not records:
                print("✅ 未发现系统类无效节点")
            else:
                print(f"⚠️ 发现 {len(records)} 个疑似系统类节点：")
                for r in records:
                    node = r["n"]
                    labels = r["labels"]
                    props = dict(node)
                    print(f"  - {labels} | id={r['id']} | name={props.get('name')}")
        driver.close()
    except Exception as e:
        print(f"❌ 查询失败: {e}")

def delete_invalid_system_nodes():
    print("\n🧹 删除系统/平台/软件类无效节点（DETACH DELETE）...")
    try:
        driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        with driver.session(database=NEO4J_DATABASE) as session:
            where_clause = _build_system_filter()
            query = f"""
                MATCH (n)
                WHERE exists(n.name) AND ({where_clause})
                WITH collect(n) as nodes
                CALL {{
                    WITH nodes
                    UNWIND nodes as n
                    DETACH DELETE n
                }}
                RETURN size(nodes) as deleted_count
            """
            res = session.run(query).single()
            count = res["deleted_count"] if res else 0
            print(f"✅ 已删除 {count} 个系统类节点")
        driver.close()
    except Exception as e:
        print(f"❌ 删除失败: {e}")

def check_status():
    print(f"Connecting to {NEO4J_URL} as {NEO4J_USERNAME}...")
    try:
        driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        driver.verify_connectivity()
        print("✅ Connection successful.")
        
        with driver.session(database=NEO4J_DATABASE) as session:
            # 1. Total Nodes
            result = session.run("MATCH (n) RETURN count(n) as count")
            node_count = result.single()["count"]
            print(f"📉 Total Nodes: {node_count}")
            
            # 2. Total Relationships
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            rel_count = result.single()["count"]
            print(f"📉 Total Relationships: {rel_count}")
            
            # 3. Node Labels Breakdown
            print("\n📊 Node Labels Breakdown:")
            result = session.run("MATCH (n) RETURN labels(n) as labels, count(*) as count ORDER BY count DESC LIMIT 10")
            for record in result:
                print(f"   - {record['labels']}: {record['count']}")
                
            # 4. Relationship Types Breakdown
            print("\n🔗 Relationship Types Breakdown:")
            result = session.run("MATCH ()-[r]->() RETURN type(r) as type, count(*) as count ORDER BY count DESC LIMIT 10")
            for record in result:
                print(f"   - {record['type']}: {record['count']}")

            # 5. Sample Data (Last 5 created nodes)
            print("\n📝 Sample Recent Nodes (limit 5):")
            result = session.run("MATCH (n) RETURN n LIMIT 5") # Ordered by insertion isn't guaranteed without timestamp, just taking 5
            for record in result:
                node = record["n"]
                labels = list(node.labels)
                props = dict(node)
                print(f"   - {labels}: {props}")

        driver.close()
        
    except Exception as e:
        print(f"❌ Error connecting or querying Neo4j: {e}")

if __name__ == "__main__":
    check_status()
    list_invalid_system_nodes(limit=25)
