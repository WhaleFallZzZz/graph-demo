"""
评估实体提取准确率的脚本
- 使用 Mock GraphStore 避免写入 Neo4j
- 使用 BAAI/bge-m3 计算向量相似度
- 生成评估报告
"""

import sys
import os
import logging
from typing import List, Dict, Any, Set
import numpy as np
from pathlib import Path
import json
import re

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables BEFORE importing llama modules
from dotenv import load_dotenv
load_dotenv()

from llama.factories import LlamaModuleFactory, ModelFactory, GraphStoreFactory, ExtractorFactory
from llama.kg_manager import KnowledgeGraphManager
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core import Document
from enhanced_entity_extractor import StandardTermMapper

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mock GraphStoreFactory to return SimplePropertyGraphStore
def mock_create_graph_store():
    logger.info("使用 Mock GraphStore (SimplePropertyGraphStore)")
    return SimplePropertyGraphStore()

# Patch the factory
GraphStoreFactory.create_graph_store = staticmethod(mock_create_graph_store)

# 标准实体数据集 (Ground Truth)
GOLD_STANDARD_ENTITIES = [
   # 疾病与问题 (Disease)
"近视", "远视", "散光", "弱视", "斜视", "病理性近视", "轴性近视", "屈光不正", "屈光参差", "老视", "并发性白内障", "后巩膜葡萄肿",
# 症状与体征 (Symptom & Sign)
"视物模糊", "眼胀", "虹视", "眼痛", "畏光", "流泪", "视力下降", "豹纹状眼底", "视网膜萎缩", "脉络膜萎缩", "黄斑出血", "漆样裂纹", "视盘杯盘比(C/D)扩大",
# 解剖结构 (Anatomy)
"角膜", "晶状体", "视网膜", "视神经", "黄斑区", "中心凹", "睫状肌", "悬韧带", "脉络膜", "巩膜", "前房", "房水",
# 检查与参数 (Examination & Parameter)
"眼轴长度", "屈光度", "远视储备", "调节幅度", "调节灵敏度", "眼压", "角膜曲率", "调节滞后", "五分记录法", "LogMAR视力表",
# 治疗与防控 (Treatment & Prevention)
"户外活动", "角膜塑形镜(OK镜)", "低浓度阿托品", "RGP镜片", "后巩膜加固术", "离焦框架镜", "视觉训练", "准分子激光手术(LASIK)", "全飞秒激光手术(SMILE)", "眼内接触镜植入(ICL)"
]

class EntityEvaluator:
    def __init__(self, gold_standard_path=None):
        self.embedding_model = None
        self.standard_embeddings = {}
        self.similarity_threshold = 0.65
        self.gold_standard_path = gold_standard_path
    
    def normalize_text(self, text: str) -> str:
        if not text:
            return text
        s = text.strip()
        # 注意：不要删除括号，因为"角膜塑形镜(OK镜)"需要括号来匹配
        # 只去除前后下划线等修饰符
        s = re.sub(r"[_`“”\"']", "", s)
        # 术语标准化映射
        s = StandardTermMapper.standardize(s)
        return s
    
    def is_numeric_or_measure(self, text: str) -> bool:
        """判断是否为纯数值或带单位的数值，评估时跳过这类非概念项"""
        if not text:
            return True
        s = text.strip()
        # 含有比较符或明显单位/数字的项视为数值型
        if re.search(r"[><≤≥]|\\d", s):
            return True
        if re.search(r"(mm|cd/m2|lx|D|度|年|cm|m|%|：)", s):
            return True
        return False
        
    def load_gold_standard(self):
        """从文本或JSON加载标准实体"""
        if self.gold_standard_path and os.path.exists(self.gold_standard_path):
            with open(self.gold_standard_path, 'r', encoding='utf-8') as f:
                # 假设文件每行一个实体
                entities = [line.strip() for line in f if line.strip()]
                return entities
        return GOLD_STANDARD_ENTITIES # 回退到默认列表

    def initialize(self):
        logger.info("正在初始化评估器...")
        
        # 初始化 Embedding 模型
        self.embedding_model = ModelFactory.create_embedding_model()
        if not self.embedding_model:
            logger.error("Embedding 模型初始化失败")
            return False
            
        entities = self.load_gold_standard()
        logger.info(f"正在计算 {len(entities)} 个标准实体的向量...")
        for entity in entities:
            # get_text_embedding 返回 list
            vec = self.embedding_model.get_text_embedding(self.normalize_text(entity))
            self.standard_embeddings[entity] = np.array(vec)
        return True

    def calculate_similarity(self, vec1, vec2):
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)

    def evaluate_entities(self, extracted_entities: Set[str]):
        results = []
        tp_count = 0
        
        logger.info(f"开始评估 {len(extracted_entities)} 个提取实体...")
        
        # 获取标准实体集合，用于快速查找
        standard_entities_set = set(self.standard_embeddings.keys())
        
        for entity in extracted_entities:
            # 获取提取实体的向量
            try:
                # 评估前先标准化，并跳过数值型/单位型项
                if self.is_numeric_or_measure(entity):
                    continue
                
                # 先标准化实体名称
                norm_entity = self.normalize_text(entity)
                
                # 优化：先检查映射后的实体是否直接在标准实体列表中
                if norm_entity in standard_entities_set:
                    # 直接匹配，相似度为1.0
                    best_match = norm_entity
                    max_sim = 1.0
                    is_match = True
                    tp_count += 1
                    results.append({
                        "entity": norm_entity,
                        "best_match": best_match,
                        "similarity": max_sim,
                        "is_match": is_match
                    })
                    continue
                
                # 如果不在标准列表中，使用向量相似度匹配
                vec = self.embedding_model.get_text_embedding(norm_entity)
                vec = np.array(vec)
            except Exception as e:
                logger.error(f"获取实体 '{entity}' 向量失败: {e}")
                continue
                
            best_match = None
            max_sim = -1.0
            
            # 与所有标准实体计算相似度
            for std_entity, std_vec in self.standard_embeddings.items():
                sim = self.calculate_similarity(vec, std_vec)
                if sim > max_sim:
                    max_sim = sim
                    best_match = std_entity
            
            is_match = max_sim >= self.similarity_threshold
            if is_match:
                tp_count += 1
                
            results.append({
                "entity": norm_entity,
                "best_match": best_match,
                "similarity": max_sim,
                "is_match": is_match
            })
            
        return results, tp_count

    def generate_report(self, results, tp_count, total_extracted):
        metrics = self.calculate_metrics(results, tp_count, total_extracted)
        
        print("\n" + "="*50)
        print("实体提取评估报告")
        print("="*50)
        print("\n[详细匹配结果]")
        print(f"{'提取实体':<25} | {'最佳匹配':<25} | {'相似度':<12} | {'结果'}")
        print("-" * 65)
        
        # 按相似度降序排列
        sorted_results = sorted(results, key=lambda x: x['similarity'], reverse=True)
        
        for res in sorted_results:
            match_icon = "✅ 匹配" if res['is_match'] else "❌ 未匹配"
            # 截断过长的字符串以保持格式整洁
            ent_str = (res['entity'][:22] + '..') if len(res['entity']) > 22 else res['entity']
            match_str = (res['best_match'][:22] + '..') if len(res['best_match']) > 22 else res['best_match']
            
            print(f"{ent_str:<25} | {match_str:<25} | {res['similarity']:.4f}     | {match_icon}")
            
        print("\n[统计指标]")
        print(f"提取实体总数: {total_extracted}")
        print(f"成功匹配数 (TP): {tp_count}")
        print(f"未匹配数: {total_extracted - tp_count}")
        print(f"准确率 (Precision): {metrics['precision']:.2%}")
        print(f"召回率 (Recall): {metrics['recall']:.2%}")
        print(f"F1 分数: {metrics['f1']:.2f}")
        print(f"标准实体覆盖率: {metrics['matched_count']}/{len(self.standard_embeddings)}")
        print(f"相似度阈值: {self.similarity_threshold}")
        print("="*50 + "\n")

    def calculate_metrics(self, results, tp_count, total_extracted):
        """计算更全面的指标"""
        precision = tp_count / total_extracted if total_extracted > 0 else 0
        
        # 召回率计算：匹配到的标准实体 / 总标准实体数
        matched_standards = set(r['best_match'] for r in results if r['is_match'])
        recall = len(matched_standards) / len(self.standard_embeddings)
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "matched_count": len(matched_standards)
        }

def main():
    # 1. 初始化管理器
    kg_manager = KnowledgeGraphManager()
    if not kg_manager.initialize():
        logger.error("KnowledgeGraphManager 初始化失败")
        return
    
    # 2. 准备测试文档
    # 从指定文件加载
    file_path = "/Users/whalefall/Documents/workspace/python_demo/青少年近视防控手册.txt"
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            test_text = f.read()
        logger.info(f"成功加载文件: {file_path}, 长度: {len(test_text)} 字符")
        # 截取前 10000 个字符进行测试，避免 token 过多导致太慢或报错
        # if len(test_text) > 10000:
        #     test_text = test_text[:10000]
        #     logger.info("文档过长，已截取前 10000 个字符进行测试")
            
        documents = [Document(text=test_text, metadata={"file_name": os.path.basename(file_path)})]
    except FileNotFoundError:
        logger.error(f"找不到文件: {file_path}，使用默认测试文本")
        test_text = """
        近视是青少年常见的眼病。眼轴长度的增加是导致轴性近视的主要原因。
        一般来说，眼轴增长率如果超过0.2mm/年，就需要引起重视。
        低浓度阿托品和角膜塑形镜是目前公认有效的近视防控手段。
        调节功能异常，如调节不足，也可能导致视疲劳。
        """
        documents = [Document(text=test_text, metadata={"file_name": "test_doc.txt"})]
    
    logger.info("开始构建知识图谱 (提取实体)...")
    
    # 3. 运行提取
    # 使用 SimplePropertyGraphStore，所以不会写入 Neo4j
    index = kg_manager.build_knowledge_graph(documents)
    
    if not index:
        logger.error("知识图谱构建失败")
        return
        
    # 4. 从 GraphStore 获取提取的实体
    # SimplePropertyGraphStore 存储在内存中
    graph_store = index.property_graph_store
    
    # 获取所有三元组
    triplets = graph_store.get_triplets()
    
    extracted_entities = set()
    for triplet in triplets:
        # triplet 是 [head, relation, tail]
        # head 和 tail 是 EntityNode
        extracted_entities.add(triplet[0].name)
        extracted_entities.add(triplet[2].name)
        
    logger.info(f"从图中提取了 {len(extracted_entities)} 个唯一实体")
    
    # 5. 评估
    evaluator = EntityEvaluator()
    if evaluator.initialize():
        results, tp_count = evaluator.evaluate_entities(extracted_entities)
        evaluator.generate_report(results, tp_count, len(extracted_entities))

if __name__ == "__main__":
    main()
