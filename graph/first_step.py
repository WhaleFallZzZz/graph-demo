import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

# 加载环境变量
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

class EyeCareKnowledgeGraph:
    def __init__(self, uri, user, password):
        # 连接数据库
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        # 关闭连接
        self.driver.close()

    def init_data(self):
        """
        初始化图谱数据：清空旧数据 -> 插入新数据
        """
        with self.driver.session() as session:
            # 1. 为了演示方便，先清空数据库中所有旧节点（生产环境慎用！）
            session.run("MATCH (n) DETACH DELETE n")
            print(">>> 旧数据已清空")

            # 2. 执行 Cypher 语句构建图谱
            # 这里我们将创建：干预手段、诊断、限制条件、风险 四类节点，并建立关系
            cypher_query = """
            // --- 创建节点 (Nodes) ---

            // 1. 干预手段 (Treatment)
            CREATE (ok:Treatment {name: '角膜塑形镜(OK镜)', type: '光学干预', price_level: '高'})
            CREATE (atropine:Treatment {name: '0.01%阿托品', type: '药物干预', price_level: '低'})
            CREATE (defocus:Treatment {name: '多点离焦框架镜', type: '光学干预', price_level: '中'})
            CREATE (surgery:Treatment {name: '近视激光手术', type: '手术干预', price_level: '高'})

            // 2. 诊断结论 (Diagnosis)
            CREATE (myopia:Diagnosis {name: '真性近视'})
            CREATE (pseudo:Diagnosis {name: '假性近视'})
            CREATE (adult:PersonType {name: '成年人'})
            CREATE (child:PersonType {name: '青少年'})

            // 3. 限制条件 (Condition - 用于逻辑推理)
            CREATE (cond_age_8:Condition {desc: '年龄>8岁'})
            CREATE (cond_deg_600:Condition {desc: '近视度数<600度'})
            CREATE (cond_curve:Condition {desc: '角膜曲率正常'})
            CREATE (cond_adult:Condition {desc: '年龄>18岁'})

            // 4. 风险与副作用 (Risk)
            CREATE (risk_light:Risk {name: '畏光'})
            CREATE (risk_inf:Risk {name: '角膜感染风险'})
            CREATE (risk_dry:Risk {name: '干眼症'})

            // --- 创建关系 (Relationships) ---

            // OK镜的逻辑：针对真性近视，有年龄和度数限制，有感染风险
            MERGE (ok)-[:SUITABLE_FOR]->(myopia)
            MERGE (ok)-[:REQUIRES]->(cond_age_8)
            MERGE (ok)-[:REQUIRES]->(cond_deg_600)
            MERGE (ok)-[:REQUIRES]->(cond_curve)
            MERGE (ok)-[:HAS_RISK]->(risk_inf)

            // 阿托品的逻辑：针对真性近视，副作用是畏光
            MERGE (atropine)-[:SUITABLE_FOR]->(myopia)
            MERGE (atropine)-[:HAS_SIDE_EFFECT]->(risk_light)

            // 离焦镜的逻辑：针对真性近视，限制较少
            MERGE (defocus)-[:SUITABLE_FOR]->(myopia)
            MERGE (defocus)-[:SUITABLE_FOR]->(child)

            // 激光手术逻辑：必须成年，且治愈近视
            MERGE (surgery)-[:SUITABLE_FOR]->(myopia)
            MERGE (surgery)-[:REQUIRES]->(cond_adult)
            MERGE (surgery)-[:HAS_RISK]->(risk_dry)
            """

            session.run(cypher_query)
            print(">>> 知识图谱构建成功！节点与关系已写入。")


if __name__ == "__main__":
    # --- 配置区域 ---
    URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    USER = os.environ.get("NEO4J_USERNAME", "neo4j")
    PASSWORD = os.environ.get("NEO4J_PASSWORD", "12345678")

    kg = EyeCareKnowledgeGraph(URI, USER, PASSWORD)
    try:
        kg.init_data()
    finally:
        kg.close()