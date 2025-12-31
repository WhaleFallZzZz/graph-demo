"""
工厂模式模块 - 负责创建核心组件
实现从配置到实例的解耦，提高代码复用性和可维护性
"""
import logging
from typing import Optional, Dict, Any

from config import API_CONFIG, EMBEDDING_CONFIG, NEO4J_CONFIG, EXTRACTOR_CONFIG, RATE_LIMIT_CONFIG, RERANK_CONFIG
from entity_extractor import parse_dynamic_triplets, MultiStageLLMExtractor
from custom_siliconflow_rerank import CustomSiliconFlowRerank

logger = logging.getLogger(__name__)

class LlamaModuleFactory:
    """LlamaIndex 模块工厂 - 单例模式管理模块导入"""
    
    _modules = None
    
    @classmethod
    def get_modules(cls):
        """延迟加载并返回所有必要的模块"""
        if cls._modules:
            return cls._modules
            
        try:
            from llama_index.core import SimpleDirectoryReader, Document, Settings, PropertyGraphIndex
            from llama_index.llms.siliconflow import SiliconFlow
            from llama_index.embeddings.siliconflow import SiliconFlowEmbedding
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
            from llama_index.core.indices.property_graph import DynamicLLMPathExtractor
            
            cls._modules = {
                'SimpleDirectoryReader': SimpleDirectoryReader,
                'Document': Document,
                'Settings': Settings,
                'PropertyGraphIndex': PropertyGraphIndex,
                'SiliconFlow': SiliconFlow,
                'SiliconFlowEmbedding': SiliconFlowEmbedding,
                'HuggingFaceEmbedding': HuggingFaceEmbedding,
                'Neo4jPropertyGraphStore': Neo4jPropertyGraphStore,
                'DynamicLLMPathExtractor': DynamicLLMPathExtractor
            }
            logger.info("成功导入所有必要的 LlamaIndex 模块")
            return cls._modules
        except ImportError as e:
            logger.error(f"❌ 导入模块失败: {e}")
            return None

class ModelFactory:
    """模型工厂 - 负责创建LLM和Embedding模型"""
    
    @staticmethod
    def create_llm():
        """创建 LLM 实例"""
        modules = LlamaModuleFactory.get_modules()
        if not modules:
            return None
            
        try:
            llm = modules['SiliconFlow'](
                api_key=API_CONFIG["siliconflow"]["api_key"],
                model=API_CONFIG["siliconflow"]["llm_model"],
                timeout=API_CONFIG["siliconflow"]["timeout"],
                max_tokens=API_CONFIG["siliconflow"]["max_tokens"],
                temperature=API_CONFIG["siliconflow"]["temperature"]
            )
            # 配置全局设置
            modules['Settings'].llm = llm
            return llm
        except Exception as e:
            logger.error(f"创建 LLM 失败: {e}")
            return None

    @staticmethod
    def create_embedding_model():
        """创建 Embedding 模型"""
        modules = LlamaModuleFactory.get_modules()
        if not modules:
            return None
            
        try:
            if EMBEDDING_CONFIG.get('use_local_model', False):
                logger.info(f"使用本地嵌入模型: {EMBEDDING_CONFIG.get('local_fallback_model')}")
                return modules['HuggingFaceEmbedding'](
                    model_name=EMBEDDING_CONFIG.get('local_fallback_model', 'BAAI/bge-m3'),
                    device=EMBEDDING_CONFIG.get('local_device', 'cpu'),
                    trust_remote_code=True
                )
            else:
                logger.info("使用在线嵌入模型: BAAI/bge-m3 via SiliconFlow")
                from custom_siliconflow_embedding import CustomSiliconFlowEmbedding
                return CustomSiliconFlowEmbedding(
                    model=API_CONFIG["siliconflow"]["embedding_model"],
                    api_key=API_CONFIG["siliconflow"]["api_key"],
                    max_retries=RATE_LIMIT_CONFIG['max_retries'],
                    request_delay=RATE_LIMIT_CONFIG['request_delay']
                )
        except Exception as e:
            logger.error(f"创建 Embedding 模型失败: {e}")
            return None

class GraphStoreFactory:
    """图存储工厂"""
    
    @staticmethod
    def create_graph_store():
        """创建 Neo4j 图存储"""
        modules = LlamaModuleFactory.get_modules()
        if not modules:
            return None
            
        try:
            return modules['Neo4jPropertyGraphStore'](
                username=NEO4J_CONFIG["username"],
                password=NEO4J_CONFIG["password"],
                url=NEO4J_CONFIG["url"],
                database=NEO4J_CONFIG["database"]
            )
        except Exception as e:
            logger.error(f"创建图存储失败: {e}")
            return None

class ExtractorFactory:
    """提取器工厂"""
    
    @staticmethod
    def create_extractor(llm, extract_prompt: str = None):
        """创建实体提取器 - 使用多阶段并行处理架构"""
        modules = LlamaModuleFactory.get_modules()
        if not modules:
            return None
            
        try:
            # 优先使用专门的实体和关系Prompt
            entity_prompt = EXTRACTOR_CONFIG.get('entity_prompt')
            relation_prompt = EXTRACTOR_CONFIG.get('relation_prompt')
            
            # 如果没有配置专门的Prompt，回退到通用的extract_prompt (兼容旧配置)
            if not entity_prompt or not relation_prompt:
                logger.warning("未检测到 entity_prompt 或 relation_prompt，回退到单阶段提取模式")
                prompt = extract_prompt or EXTRACTOR_CONFIG['extract_prompt']
                return modules['DynamicLLMPathExtractor'](
                    llm=llm,
                    extract_prompt=prompt,
                    parse_fn=parse_dynamic_triplets,
                    num_workers=EXTRACTOR_CONFIG.get('num_workers', 4),
                    max_triplets_per_chunk=EXTRACTOR_CONFIG['max_triplets_per_chunk']
                )

            # 使用多阶段并行提取器
            import os
            # 默认使用CPU核心数的一半作为并发数，防止过载
            default_workers = max(2, (os.cpu_count() or 4) // 2)
            num_workers = EXTRACTOR_CONFIG.get('num_workers', default_workers)
            
            logger.info(f"初始化多阶段LLM提取器，并发工作线程数: {num_workers}")
            
            # 获取图存储实例用于流式写入
            graph_store = GraphStoreFactory.create_graph_store()
            
            return MultiStageLLMExtractor(
                llm=llm,
                entity_prompt=entity_prompt,
                relation_prompt=relation_prompt,
                num_workers=num_workers,
                max_triplets_per_chunk=EXTRACTOR_CONFIG['max_triplets_per_chunk'],
                graph_store=graph_store
            )
        except Exception as e:
            logger.error(f"创建提取器失败: {e}")
            return None

class RerankerFactory:
    """重排序器工厂"""
    
    @staticmethod
    def create_reranker():
        """创建重排序器"""
        if not RERANK_CONFIG.get("enable", False):
            return None
            
        try:
            provider = RERANK_CONFIG.get("provider", "siliconflow")
            
            if provider == "siliconflow":
                return CustomSiliconFlowRerank(
                    api_key=RERANK_CONFIG["api_key"],
                    model=RERANK_CONFIG["model"],
                    top_n=RERANK_CONFIG["top_n"]
                )
            
            return None
        except Exception as e:
            logger.error(f"创建重排序器失败: {e}")
            return None
