import os
import re
from openai import OpenAI
from neo4j import GraphDatabase
from dotenv import load_dotenv

# 加载环境变量
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

# ================= 配置区域 =================
SILICON_API_KEY = os.environ.get("SILICONFLOW_API_KEY", "sk-cjfdzvbbpuncpnzkydpxivhchhaiblgvlydzfupsrnohoxja")
SILICON_BASE_URL = os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
MODEL_NAME = os.environ.get("GRAPH_RAG_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "12345678")


# ===========================================

class Neo4jGraphRAG:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self.client = OpenAI(api_key=SILICON_API_KEY, base_url=SILICON_BASE_URL)
        # 初始化时自动获取数据库结构，无需手动修改 Prompt
        self.schema_str = self._get_db_schema()

    def close(self):
        self.driver.close()

    def _get_db_schema(self):
        """
        🔥 核心优化 1: 动态获取 Schema
        自动读取数据库里的所有节点标签和关系类型，解决硬编码跟不上数据库变化的问题。
        """
        schema_info = []
        try:
            with self.driver.session() as session:
                # 获取所有节点标签
                result_nodes = session.run("CALL db.labels()")
                labels = [record["label"] for record in result_nodes]
                schema_info.append(f"包含的节点类型 (Labels): {', '.join(labels)}")

                # 获取所有关系类型
                result_rels = session.run("CALL db.relationshipTypes()")
                rels = [record["relationshipType"] for record in result_rels]
                schema_info.append(f"包含的关系类型 (Relationships): {', '.join(rels)}")

                # (可选) 获取部分属性示例，帮助 AI 理解 name 还是 title
                # 这里简单处理，假设主要属性是 name
        except Exception as e:
            print(f"⚠️ 获取 Schema 失败，将使用默认 Schema: {e}")
            return "节点: Treatment, Risk, Condition\n关系: HAS_RISK, REQUIRES"

        return "\n".join(schema_info)

    def _clean_content(self, content):
        """
        🔥 核心优化 2: 强力清洗 DeepSeek 的思考过程
        """
        # 1. 移除 <think> 标签及其内容 (支持多行，支持未闭合的情况)
        # 这种正则策略可以防止 <think> 只有开头没有结尾导致吞掉后面内容的情况
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<think>.*', '', content, flags=re.DOTALL | re.IGNORECASE)  # 兜底：去掉未闭合的

        # 2. 提取代码块，兼容 ```cypher, ```sql, 或者只有 ``` 的情况
        code_match = re.search(r"```(?:cypher|sql)?\s*(.*?)\s*```", content, re.DOTALL | re.IGNORECASE)
        if code_match:
            code = code_match.group(1)
        else:
            code = content  # 如果没写代码块，尝试直接用全文

        # 3. 去掉注释和空行
        lines = [line for line in code.split('\n') if line.strip() and not line.strip().startswith("//")]
        return '\n'.join(lines).strip()

    def text_to_cypher(self, user_question):
        print(f"🤖 (Schema已加载) 正在生成查询: {user_question} ...")

        system_prompt = f"""
        你是一名 Neo4j 图数据库专家。请根据用户问题和给定的数据库 Schema 生成 Cypher 查询语句。

        【当前数据库 Schema (实时获取)】:
        {self.schema_str}

        【Schema】:
        节点: Treatment, Risk, Condition, Diagnosis
        关系: [:HAS_RISK], [:REQUIRES], [:SUITABLE_FOR], [:HAS_SIDE_EFFECT]
        
        【生成规则】:
        1. 🎯 **模糊匹配**: 用户输入的可能是简称（如"OK镜"），数据库可能存的是全称。请务必使用 `CONTAINS`进行查询。
           例如: `WHERE t.name =~ 'OK镜'`
        2. 🚫 **禁止臆造**: 只能使用 Schema 中列出的节点和关系类型。
        3. ⚡ **方向准确**: 注意关系的箭头方向，通常是 `(主语)-[:动作]->(宾语)`。

        【参考示例 (Few-Shot)】:
        用户: "阿托品适合什么人？"
        回答:
        ```cypher
        MATCH (t:Treatment)-[:SUITABLE_FOR]->(d)
        WHERE t.name CONTAINS '阿托品'
        RETURN t.name, labels(d), d.name
        ```

        用户: "近视手术有什么风险？"
        回答:
        ```cypher
        MATCH (t:Treatment)-[:HAS_RISK]->(r)
        WHERE t.name CONTAINS '近视' AND t.name CONTAINS '手术'
        RETURN t.name, r.name
        ```

        请直接输出 Cypher 代码块。
        """

        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_question}
                ],
                temperature=0.1,
                max_tokens=1024  # 稍微调大一点防止截断
            )
            raw_content = response.choices[0].message.content
            return self._clean_content(raw_content)
        except Exception as e:
            print(f"❌ LLM 调用失败: {e}")
            return None

    def execute_cypher(self, query):
        if not query: return []
        results = []
        try:
            with self.driver.session() as session:
                print(f"🔍 执行 SQL: {query}")
                result = session.run(query)
                # 优化结果格式化：尝试获取更有意义的字段
                for record in result:
                    # 将每行记录转换为字符串，方便后续 RAG 处理
                    results.append(" | ".join([str(val) for val in record.values()]))
        except Exception as e:
            print(f"❌ Cypher 执行报错: {e}")
            # 高级优化: 这里其实可以把错误扔回给 LLM 让它重写 (Self-Healing)，暂时先略过
        return list(set(results))

    def generate_answer(self, user_question, db_data):
        if not db_data:
            return "抱歉，知识库中暂时没有找到相关信息。"

        prompt = f"""
        你是一名专业的眼科医生助手。基于以下数据库检索到的事实回答患者问题。

        【用户问题】: {user_question}
        【数据库事实】:
        {db_data}

        【要求】:
        1. 语气亲切、专业。
        2. 如果事实中包含列表（如多个副作用），请分点陈述。
        3. 不要编造数据库中不存在的信息。
        """

        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return self._clean_content(response.choices[0].message.content)


# ================= 运行测试 =================
if __name__ == "__main__":
    app = Neo4jGraphRAG()

    # 你的测试问题
    questions = [
        "我 6 岁，能戴 OK 镜吗？",
        # "做飞秒手术有什么要求？"
    ]

    for q in questions:
        print(f"\n======== 处理问题: {q} ========")
        cypher_sql = app.text_to_cypher(q)
        if cypher_sql:
            data = app.execute_cypher(cypher_sql)
            print(f"📄 检索结果: {data}")
            final_ans = app.generate_answer(q, data)
            print(f"👩‍⚕️ 最终回复:\n{final_ans}")

    app.close()