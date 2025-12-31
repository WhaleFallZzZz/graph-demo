import os
import json
import sys
import logging
from typing import List, Dict, Any
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from neo4j import GraphDatabase
except ImportError:
    print("Error: neo4j library is not installed. Please install it using 'pip install neo4j'.")
    sys.exit(1)

from llama.config import NEO4J_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_valid_label(labels: List[str]) -> str:
    """
    Filter out system labels and return the most appropriate label.
    """
    # System labels to ignore (LlamaIndex specific and common generic ones)
    IGNORED_LABELS = {'__Entity__', 'Entity', '__Vector__', '__Embedding__', 'Resource', '_Entity_', '__Node__', 'Chunk'}
    
    valid_labels = [l for l in labels if l not in IGNORED_LABELS]
    
    if valid_labels:
        return valid_labels[0]
    return "Unknown"

def export_neo4j_to_json(output_file: str = "neo4j_export.json"):
    """
    Export all entities and relations from Neo4j to a JSON file.
    """
    uri = NEO4J_CONFIG.get('url') or os.environ.get("NEO4J_URL", "bolt://101.43.120.88:7687")
    user = NEO4J_CONFIG.get('username') or os.environ.get("NEO4J_USERNAME", "neo4j")
    password = NEO4J_CONFIG.get('password') or os.environ.get("NEO4J_PASSWORD", "J.pQF!zg33haagT")
    database = NEO4J_CONFIG.get('database') or os.environ.get("NEO4J_DATABASE", "neo4j")

    if not uri or not user or not password:
        logger.error("Neo4j configuration is missing. Please check your config.py or environment variables.")
        return

    logger.info(f"Connecting to Neo4j at {uri}...")
    
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        
        # Verify connection
        driver.verify_connectivity()
        logger.info("Connected successfully.")

        query = """
        MATCH (h)-[r]->(t)
        RETURN h.name as head, labels(h) as head_labels, type(r) as relation, t.name as tail, labels(t) as tail_labels
        """

        results = []
        
        with driver.session(database=database) as session:
            logger.info("Executing Cypher query to fetch all relations...")
            # Using run() fetches all results. For very large datasets, one might consider streaming or pagination.
            # Here we stream using the iterator.
            result = session.run(query)
            
            count = 0
            for record in result:
                head_name = record['head']
                tail_name = record['tail']
                
                # Skip if names are missing (unlikely for EntityNodes but possible for other nodes)
                if not head_name or not tail_name:
                    continue

                head_type = get_valid_label(record['head_labels'])
                tail_type = get_valid_label(record['tail_labels'])
                relation_type = record['relation']

                results.append({
                    "head": head_name,
                    "head_type": head_type,
                    "relation": relation_type,
                    "tail": tail_name,
                    "tail_type": tail_type
                })
                count += 1
                
                if count % 1000 == 0:
                    logger.info(f"Processed {count} relations...")

        logger.info(f"Total relations fetched: {len(results)}")

        # Write to JSON
        logger.info(f"Writing results to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Export completed successfully. File saved at: {os.path.abspath(output_file)}")

    except Exception as e:
        logger.error(f"An error occurred during export: {e}")
    finally:
        if 'driver' in locals():
            driver.close()

if __name__ == "__main__":
    export_neo4j_to_json()
