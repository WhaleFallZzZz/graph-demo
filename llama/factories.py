"""
工厂模式模块 - 负责创建核心组件
实现从配置到实例的解耦，提高代码复用性和可维护性
"""
import logging
from typing import Optional, Dict, Any

from config import API_CONFIG, EMBEDDING_CONFIG, NEO4J_CONFIG, EXTRACTOR_CONFIG, RATE_LIMIT_CONFIG, RERANK_CONFIG, DOCUMENT_CONFIG
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
            from llama_index.core.graph_stores import SimplePropertyGraphStore
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
                'SimplePropertyGraphStore': SimplePropertyGraphStore,
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
            # 使用配置中的分块大小，与文档分块策略保持一致
            modules['Settings'].chunk_size = DOCUMENT_CONFIG.get('chunk_size', 600)
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
                return modules['HuggingFaceEmbedding'](
                    model_name=EMBEDDING_CONFIG.get('local_fallback_model', 'BAAI/bge-m3'),
                    device=EMBEDDING_CONFIG.get('local_device', 'cpu'),
                    trust_remote_code=True
                )
            from custom_siliconflow_embedding import CustomSiliconFlowEmbedding
            try:
                logger.info("尝试使用在线嵌入模型 via SiliconFlow")
                return CustomSiliconFlowEmbedding(
                    model=API_CONFIG["siliconflow"]["embedding_model"],
                    api_key=API_CONFIG["siliconflow"]["api_key"],
                    max_retries=RATE_LIMIT_CONFIG['max_retries'],
                    request_delay=RATE_LIMIT_CONFIG['request_delay']
                )
            except Exception as e:
                logger.warning(f"在线嵌入模型不可用，回退到本地模型: {e}")
                return modules['HuggingFaceEmbedding'](
                    model_name=EMBEDDING_CONFIG.get('local_fallback_model', 'BAAI/bge-m3'),
                    device=EMBEDDING_CONFIG.get('local_device', 'cpu'),
                    trust_remote_code=True
                )
        except Exception as e:
            logger.error(f"创建 Embedding 模型失败: {e}")
            return None
    
    @staticmethod
    def create_lightweight_llm():
        """创建轻量级LLM实例（用于三元组反向校验）"""
        modules = LlamaModuleFactory.get_modules()
        if not modules:
            return None
            
        try:
            # 优先使用配置的轻量级模型，如果没有配置则使用默认的 Qwen/Qwen2.5-7B-Instruct
            lightweight_model = API_CONFIG["siliconflow"].get("lightweight_model") or "Qwen/Qwen2.5-7B-Instruct"
            
            lightweight_llm = modules['SiliconFlow'](
                api_key=API_CONFIG["siliconflow"]["api_key"],
                model=lightweight_model,
                timeout=API_CONFIG["siliconflow"]["timeout"],
                max_tokens=2048,  # 轻量级模型使用较小的max_tokens
                temperature=0.0  # 校验任务需要确定性输出
            )
            logger.info(f"创建轻量级校验模型成功: {lightweight_model}")
            return lightweight_llm
        except Exception as e:
            logger.error(f"创建轻量级LLM失败: {e}")
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
            username = NEO4J_CONFIG.get("username")
            password = NEO4J_CONFIG.get("password")
            url = NEO4J_CONFIG.get("url")
            database = NEO4J_CONFIG.get("database")
            if not username or not password or not url or not database:
                logger.warning("Neo4j 配置不完整，回退到内存图存储 SimplePropertyGraphStore")
                return modules['SimplePropertyGraphStore']()
            return modules['Neo4jPropertyGraphStore'](
                username=username,
                password=password,
                url=url,
                database=database
            )
        except Exception as e:
            logger.error(f"创建图存储失败: {e}")
            try:
                logger.warning("回退到内存图存储 SimplePropertyGraphStore")
                return modules['SimplePropertyGraphStore']()
            except Exception as e2:
                logger.error(f"创建内存图存储失败: {e2}")
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
            prompt = extract_prompt or EXTRACTOR_CONFIG['extract_prompt']
            return modules['DynamicLLMPathExtractor'](
                llm=llm,
                extract_prompt=prompt,
                parse_fn=parse_dynamic_triplets,
                num_workers=EXTRACTOR_CONFIG.get('num_workers', 4),
                max_triplets_per_chunk=EXTRACTOR_CONFIG['max_triplets_per_chunk']
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
