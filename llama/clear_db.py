import os
from dotenv import load_dotenv
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

# 加载环境变量
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

graph_store = Neo4jPropertyGraphStore(
    username=os.environ.get("NEO4J_USERNAME", "neo4j"),
    password=os.environ.get("NEO4J_PASSWORD", "12345678"),
    url=os.environ.get("NEO4J_URL", "bolt://localhost:7687"),
)

# Execute a query to delete all nodes and relationships
# Note: Neo4jPropertyGraphStore doesn't have a direct 'clear' method exposed simply, 
# so we use the structured_query method or underlying driver if accessible.
# However, structured_query returns data.
# A simple way is to use the driver directly.

driver = graph_store._driver
with driver.session() as session:
    session.run("MATCH (n) DETACH DELETE n")

print("Database cleared.")
