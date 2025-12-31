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
MODEL_CC = os.environ.get("GRAPH_RAG_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
MODEL_NAME = os.environ.get("SILICONFLOW_CODER_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")
# MODEL_CHAT = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "12345678")  # ⚠️ 填入数据库密码
# ===========================================

client = OpenAI(
    api_key=SILICON_API_KEY,
    base_url=SILICON_BASE_URL
)


def extract_cypher_code(content):
    """
    清洗函数保持不变：提取代码块
    """
    code_block_pattern = r"```(?:cypher)?\s*(.*?)\s*```"
    match = re.search(code_block_pattern, content, re.DOTALL | re.IGNORECASE)
    if match:
        code = match.group(1)
    else:
        # 清洗 think 标签 (支持大小写，支持未闭合的情况)
        code = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE)
        code = re.sub(r'<think>.*', '', code, flags=re.DOTALL | re.IGNORECASE)

    # 过滤非代码行
    lines = code.split('\n')
    valid_lines = [line for line in lines if not line.strip().startswith("//")]
    return '\n'.join(valid_lines).strip()


def text_to_cypher(user_question):
    print(f"🤖 正在思考: {user_question} ...")

    # 🌟 V3 核心改动：加入 Few-Shot Examples (示例教学)
    system_prompt = f"""
    你是一名 Neo4j 专家。将用户问题转换为 Cypher 查询。

    【Schema】:
    节点: Treatment, Risk, Condition, Diagnosis
    关系: [:HAS_RISK], [:REQUIRES], [:SUITABLE_FOR], [:HAS_SIDE_EFFECT]

    【必读：正确示例】:
    用户: "OK镜有什么风险？"
    你的回答: 
    ```cypher
    MATCH (t:Treatment)-[:HAS_RISK]->(r:Risk) 
    WHERE t.name CONTAINS 'OK镜' 
    RETURN r.name
    ```

    用户: "阿托品适合谁？"
    你的回答:
    ```cypher
    MATCH (t:Treatment)-[:SUITABLE_FOR]->(d:Diagnosis)
    WHERE t.name CONTAINS '阿托品'
    RETURN t.name, d.name
    ```
    
    用户: "做飞秒手术有什么要求？"
    你的回答:
    ```cypher
    MATCH (t:Treatment)-[:REQUIRES]->(c:Condition)
    WHERE t.name CONTAINS '飞秒'
    RETURN t.name, c.name
    ```
    
    用户: "我 6 岁，能戴 OK 镜吗？"
    你的回答:
    ```cypher
    MATCH (t:Treatment)-[:REQUIRES]->(c:Condition)
    WHERE t.name CONTAINS 'OK镜'
    RETURN t.name, c.name
    ```

    【严禁事项】:
    1. ❌ 严禁使用 `(a)-(b)` 这种单横线语法！必须使用完整箭头 `(a)-[:RELATION_TYPE]->(b)`。
    2. ❌ 严禁在 {{}} 中直接赋值 name。必须使用 `WHERE t.name CONTAINS '...'`。
    3. 只输出代码块，不要废话。
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question}
            ],
            temperature=0.3,  # 低温保证稳定
            max_tokens=4096
        )

        raw_content = response.choices[0].message.content
        return extract_cypher_code(raw_content)

    except Exception as e:
        print(f"❌ API 失败: {e}")
        return None


def execute_cypher(query):
    if not query: return []
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    results = []
    try:
        with driver.session() as session:
            print(f"🔍 执行 Cypher: {query}")
            result = session.run(query)
            for record in result:
                results.extend([str(val) for val in record.values()])
    except Exception as e:
        print(f"❌ 数据库报错: {e}")
    finally:
        driver.close()
    return list(set(results))


def generate_final_answer(user_question, db_data):
    """
        RAG 的最后一步：生成 (Generation) - 严格防幻觉版
        """
    # 1. 如果没查到数据，直接返回，不给 LLM 编造的机会
    if not db_data or db_data == []:
        return "抱歉，根据目前的眼科知识库，没有查到关于此问题的记录。"

    # 2. 强力清洗 Prompt：要求“不懂就闭嘴”
    prompt = f"""
        你是一个严谨的眼科医学助手。
        用户的提问："{user_question}"

        【知识库检索结果】（这是唯一可信的事实来源）：
        {db_data}

        【回答要求】:
        1. ⚠️ **严禁** 使用你预训练的外部知识来解释药物成分或功能（防止幻觉）。
        2. 只能根据【检索结果】进行简单的语义连接。
        3. 如果检索结果只包含名词（如['真性近视']），就只回答该名词，不要展开解释它是什么。
        4. 语气要专业、简洁。
        """

    # print(f"💬 正在组织语言...")
    response = client.chat.completions.create(
        model=MODEL_CC,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.6
    )

    # 同样记得清洗掉 <think> 标签
    raw = response.choices[0].message.content
    # 3. 再次强力清洗 <think> 标签 (防止漏网之鱼)
    clean_text = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL | re.IGNORECASE)
    clean_text = re.sub(r'<think>.*', '', clean_text, flags=re.DOTALL | re.IGNORECASE) # 处理截断
    return clean_text.strip()

if __name__ == "__main__":
    # 再次测试
    # q1 = "OK镜有什么风险？"
    # q1 = "阿托品适合谁？"
    # q1 = "OK镜会导致什么后果？"
    # q1 = "做飞秒手术有什么要求？"
    # q1 = "我 6 岁，能戴 OK 镜吗？"
    q1 = "近视激光手术有什么风险？"
    ans1 = execute_cypher(text_to_cypher(q1))
    print(f"✅ 答案: {ans1}\n")
    final_speech = generate_final_answer(q1, ans1)
    print(f"👩‍⚕️ 最终回复: {final_speech}")
