import pandas as pd
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

# ================= 配置区域 =================
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "12345678")  # ⚠️ 记得修改密码
EXCEL_FILE = "eye_care_data.xlsx"


# ===========================================

def generate_mock_excel():
    """
    生成一个模拟的 Excel 文件，包含两张表：
    1. Treatments: 所有的干预手段（核心节点）
    2. Relations: 所有的逻辑关系（知识的核心）
    """
    print(f"📊 正在生成演示数据文件: {EXCEL_FILE} ...")

    # 1. 干预手段表 (Nodes)
    treatments_data = [
        {"name": "角膜塑形镜(OK镜)", "type": "光学干预", "desc": "夜戴型硬性透气隐形眼镜"},
        {"name": "0.01%阿托品", "type": "药物干预", "desc": "低浓度抗胆碱药物"},
        {"name": "星趣控(Essilor)", "type": "离焦框架镜", "desc": "依视路微透镜设计"},
        {"name": "新乐学(Hoya)", "type": "离焦框架镜", "desc": "豪雅多点近视离焦"},
        {"name": "MiSight软镜", "type": "离焦软镜", "desc": "库博日抛型离焦软镜"},
        {"name": "飞秒激光手术", "type": "屈光手术", "desc": "成年人近视矫正手术"}
    ]

    # 2. 关系表 (Edges) - 这是知识图谱的灵魂！
    # 格式：源头(产品) -> 关系类型 -> 目标(具体的条件/风险/适应症) -> 目标的类型(Label)
    relations_data = [
        # --- OK镜的数据 ---
        {"source": "角膜塑形镜(OK镜)", "rel": "SUITABLE_FOR", "target": "真性近视", "target_label": "Diagnosis"},
        {"source": "角膜塑形镜(OK镜)", "rel": "REQUIRES", "target": "年龄>8岁", "target_label": "Condition"},
        {"source": "角膜塑形镜(OK镜)", "rel": "REQUIRES", "target": "近视<600度", "target_label": "Condition"},
        {"source": "角膜塑形镜(OK镜)", "rel": "REQUIRES", "target": "散光<150度", "target_label": "Condition"},
        {"source": "角膜塑形镜(OK镜)", "rel": "HAS_RISK", "target": "角膜感染风险", "target_label": "Risk"},
        {"source": "角膜塑形镜(OK镜)", "rel": "HAS_SIDE_EFFECT", "target": "干眼症", "target_label": "Risk"},

        # --- 阿托品的数据 ---
        {"source": "0.01%阿托品", "rel": "SUITABLE_FOR", "target": "真性近视", "target_label": "Diagnosis"},
        {"source": "0.01%阿托品", "rel": "SUITABLE_FOR", "target": "假性近视", "target_label": "Diagnosis"},
        {"source": "0.01%阿托品", "rel": "SUITABLE_FOR", "target": "眼轴增长过快", "target_label": "Diagnosis"},
        {"source": "0.01%阿托品", "rel": "HAS_SIDE_EFFECT", "target": "畏光(瞳孔散大)", "target_label": "Risk"},
        {"source": "0.01%阿托品", "rel": "HAS_SIDE_EFFECT", "target": "看近模糊", "target_label": "Risk"},

        # --- 星趣控/新乐学 (离焦镜) ---
        {"source": "星趣控(Essilor)", "rel": "SUITABLE_FOR", "target": "青少年", "target_label": "PersonType"},
        {"source": "星趣控(Essilor)", "rel": "SUITABLE_FOR", "target": "真性近视", "target_label": "Diagnosis"},
        {"source": "星趣控(Essilor)", "rel": "REQUIRES", "target": "全天佩戴", "target_label": "Condition"},

        # --- 飞秒激光 ---
        {"source": "飞秒激光手术", "rel": "REQUIRES", "target": "年龄>18岁", "target_label": "Condition"},
        {"source": "飞秒激光手术", "rel": "REQUIRES", "target": "度数稳定2年以上", "target_label": "Condition"},
        {"source": "飞秒激光手术", "rel": "HAS_RISK", "target": "干眼症", "target_label": "Risk"},
        {"source": "飞秒激光手术", "rel": "HAS_RISK", "target": "夜间眩光", "target_label": "Risk"}
    ]

    with pd.ExcelWriter(EXCEL_FILE) as writer:
        pd.DataFrame(treatments_data).to_excel(writer, sheet_name='Treatments', index=False)
        pd.DataFrame(relations_data).to_excel(writer, sheet_name='Relations', index=False)

    print("✅ Excel 文件生成完毕！")


def import_to_neo4j():
    """
    读取 Excel 并写入 Neo4j
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    # 读取 Excel
    df_treatments = pd.read_excel(EXCEL_FILE, sheet_name='Treatments')
    df_relations = pd.read_excel(EXCEL_FILE, sheet_name='Relations')

    print(f"🚀 开始导入数据到 Neo4j...")

    with driver.session() as session:
        # 1. 清空旧数据 (开发阶段为了防止重复，先清空)
        session.run("MATCH (n) DETACH DELETE n")
        print("   🧹 旧数据已清空")

        # 2. 创建核心节点 (Treatment)
        for index, row in df_treatments.iterrows():
            cypher = """
            MERGE (t:Treatment {name: $name})
            SET t.type = $type, t.desc = $desc
            """
            session.run(cypher, name=row['name'], type=row['type'], desc=row['desc'])
        print(f"   📦 已导入 {len(df_treatments)} 个核心产品节点")

        # 3. 创建关系和目标节点 (这是最重要的一步)
        # 逻辑：对于每一行关系，先找到源节点，再创建目标节点(如果不存在)，最后连线
        count = 0
        for index, row in df_relations.iterrows():
            # 动态构建 Cypher，因为 Target 的 Label (Risk/Condition) 是变化的
            target_label = row['target_label']
            rel_type = row['rel']

            # 使用 f-string 构建动态标签 (Cypher 不支持参数化 Label，所以只能拼字符串，注意安全)
            cypher = f"""
            MATCH (source:Treatment {{name: $source_name}})
            MERGE (target:{target_label} {{name: $target_name}})
            MERGE (source)-[:{rel_type}]->(target)
            """

            session.run(cypher, source_name=row['source'], target_name=row['target'])
            count += 1

        print(f"   🔗 已建立 {count} 条逻辑关系")

    driver.close()
    print("🎉 恭喜！知识图谱构建完成！")


if __name__ == "__main__":
    # 1. 只要你没有这个excel，我就帮你生成一个
    if not os.path.exists(EXCEL_FILE):
        generate_mock_excel()

    # 2. 执行导入
    import_to_neo4j()