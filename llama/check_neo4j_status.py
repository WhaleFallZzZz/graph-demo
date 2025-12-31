import os
import sys
from neo4j import GraphDatabase

# Mimic the config loading
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "12345678")
NEO4J_URL = os.environ.get("NEO4J_URL", "bolt://localhost:7687")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")

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
