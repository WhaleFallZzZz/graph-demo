import json
import logging
import sys
import os
from neo4j import GraphDatabase

# Ensure project root is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llama.config import NEO4J_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def restore_nodes():
    if not os.path.exists("deleted_nodes_backup.json"):
        logger.error("No backup file found.")
        return

    with open("deleted_nodes_backup.json", "r", encoding="utf-8") as f:
        nodes = json.load(f)

    # Only restore nodes deleted by Semantic Filter
    # Reason contains "Low Semantic Score"
    to_restore = [n for n in nodes if "Semantic" in n.get("reason", "")]
    
    logger.info(f"Found {len(to_restore)} nodes to restore (deleted by Semantic Filter).")
    
    if not to_restore:
        return

    driver = GraphDatabase.driver(
        NEO4J_CONFIG["url"],
        auth=(NEO4J_CONFIG["username"], NEO4J_CONFIG["password"])
    )

    with driver.session() as session:
        count = 0
        for node in to_restore:
            # We restore them with Entity label. 
            # Note: Relationships are lost.
            session.run("MERGE (n:Entity {name: $name})", name=node["name"])
            count += 1
            if count % 100 == 0:
                logger.info(f"Restored {count} nodes...")

    logger.info(f"Successfully restored {count} nodes.")
    driver.close()

if __name__ == "__main__":
    restore_nodes()
