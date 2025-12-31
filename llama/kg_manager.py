#!/usr/bin/env python3
"""
知识图谱管理器 - 核心业务逻辑
使用工厂模式重构，负责协调各个组件的工作
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
from concurrent.futures import ThreadPoolExecutor

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from config import setup_logging, DOCUMENT_CONFIG, API_CONFIG, EMBEDDING_CONFIG, NEO4J_CONFIG, OSS_CONFIG, RERANK_CONFIG
from factories import LlamaModuleFactory, ModelFactory, GraphStoreFactory, ExtractorFactory, RerankerFactory
from progress_sse import ProgressTracker, progress_callback
from oss_uploader import COSUploader, OSSConfig
from ocr_parser import DeepSeekOCRParser

# 设置日志
logger = setup_logging()

class KnowledgeGraphManager:
    """知识图谱管理器 - 核心Facade"""
    
    def __init__(self):
        self.modules = None
        self.llm = None
        self.embed_model = None
        self.graph_store = None
        self.executor = ThreadPoolExecutor(max_workers=3)
        self._initialized = False
        
    def initialize(self) -> bool:
        """初始化所有组件"""
        try:
            if self._initialized:
                return True
                
            progress_callback("initialization", "正在初始化知识图谱管理器...")
            
            # 1. 加载模块
            self.modules = LlamaModuleFactory.get_modules()
            if not self.modules:
                progress_callback("initialization", "模块导入失败", 0)
                return False
                
            # 2. 创建LLM
            progress_callback("initialization", "正在初始化LLM模型...", 20)
            self.llm = ModelFactory.create_llm()
            if not self.llm:
                progress_callback("initialization", "LLM初始化失败", 0)
                return False
            
            # 3. 创建Embedding
            progress_callback("initialization", "正在初始化嵌入模型...", 40)
            self.embed_model = ModelFactory.create_embedding_model()
            if not self.embed_model:
                progress_callback("initialization", "Embedding初始化失败", 0)
                return False
            
            # 4. 创建图存储
            progress_callback("initialization", "正在初始化图数据库...", 60)
            self.graph_store = GraphStoreFactory.create_graph_store()
            if not self.graph_store:
                progress_callback("initialization", "图存储初始化失败", 0)
                return False
            
            # 测试连接
            progress_callback("initialization", "正在测试数据库连接...", 80)
            try:
                self.graph_store.structured_query("MATCH (n) RETURN count(n) LIMIT 1")
                logger.info("Neo4j连接测试成功")
            except Exception as e:
                logger.warning(f"Neo4j连接测试失败: {e}")
                # 不中断，继续执行，因为可能是网络波动
            
            progress_callback("initialization", "初始化完成", 100)
            self._initialized = True
            logger.info("✅ 知识图谱管理器初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"初始化失败: {e}")
            progress_callback("initialization", f"初始化失败: {str(e)}", 0)
            return False
    
    def load_documents(self, progress_tracker: Optional[ProgressTracker] = None) -> list:
        """加载文档"""
        try:
            if not self.modules:
                self.initialize()
                
            if progress_tracker:
                progress_tracker.update_stage("document_loading", "正在加载文档...")
            else:
                progress_callback("document_loading", "正在加载文档...", 10)
            
            reader = self.modules['SimpleDirectoryReader'](
                input_dir=DOCUMENT_CONFIG['path'],
                required_exts=DOCUMENT_CONFIG.get('supported_extensions', ['.txt', '.docx', '.pdf']),
                recursive=True,
                encoding='utf-8',
                file_extractor={".pdf": DeepSeekOCRParser()}
            )
            
            documents = reader.load_data()
            
            msg = f"成功加载 {len(documents)} 个文档"
            if progress_tracker:
                progress_tracker.update_stage("document_loading", msg)
            else:
                progress_callback("document_loading", msg, 15)
                
            logger.info(f"✅ {msg}")
            return documents
            
        except Exception as e:
            error_msg = f"加载文档失败: {e}"
            logger.error(error_msg)
            if progress_tracker:
                progress_tracker.error("document_loading", error_msg)
            else:
                progress_callback("document_loading", error_msg, 0)
            return []
    
    def build_knowledge_graph(self, documents: list, progress_tracker: Optional[ProgressTracker] = None) -> Any:
        """构建知识图谱"""
        if not documents:
            error_msg = "没有文档可用于构建知识图谱"
            if progress_tracker:
                progress_tracker.error("knowledge_graph", error_msg)
            return None
        
        try:
            if progress_tracker:
                progress_tracker.update_stage("knowledge_graph", "开始构建知识图谱...")
            else:
                progress_callback("knowledge_graph", "开始构建知识图谱...", 20)
            
            # 创建提取器
            if progress_tracker:
                progress_tracker.update_stage("knowledge_graph", "正在创建实体提取器...", 25)
                
            extractor = ExtractorFactory.create_extractor(self.llm)
            if not extractor:
                error_msg = "实体提取器创建失败"
                if progress_tracker:
                    progress_tracker.error("knowledge_graph", error_msg)
                return None
            
            # 进度显示函数
            total_docs = len(documents)
            def progress_hook(completed: int, total: int, description: str = ""):
                percentage = 30 + (completed / total) * 60
                msg = f"{description} ({completed}/{total})"
                if progress_tracker:
                    progress_tracker.update_progress("knowledge_graph", msg, percentage=percentage)
                else:
                    progress_callback("knowledge_graph", msg, percentage)
            
            if progress_tracker:
                progress_tracker.update_stage("knowledge_graph", "正在初始化图谱索引...", 30)
            
            # 初始化空索引 (仅建立管道)
            # 使用空列表初始化，PropertyGraphIndex 会设置好 store 和 extractors
            index = self.modules['PropertyGraphIndex'].from_documents(
                [],
                llm=self.llm,
                embed_model=self.embed_model,
                property_graph_store=self.graph_store,
                kg_extractors=[extractor],
                show_progress=False
            )
            
            # 迭代插入文档并估算时间
            import time
            start_time = time.time()
            
            logger.info(f"开始处理 {total_docs} 个文档...")
            
            for i, doc in enumerate(documents):
                # 估算剩余时间
                elapsed = time.time() - start_time
                if i > 0:
                    avg_time = elapsed / i
                    remaining_docs = total_docs - i
                    eta_seconds = int(avg_time * remaining_docs)
                    
                    if eta_seconds > 60:
                        eta_str = f"预计剩余 {eta_seconds // 60}分{eta_seconds % 60}秒"
                    else:
                        eta_str = f"预计剩余 {eta_seconds}秒"
                else:
                    eta_str = "正在估算时间..."
                
                # 更新进度 (开始处理)
                progress_hook(i, total_docs, f"正在处理: {eta_str}")
                
                # 插入文档 (耗时操作)
                index.insert(doc)
                
                # 更新进度 (完成处理)
                progress_hook(i + 1, total_docs, f"完成文档 {i+1}/{total_docs}")
            
            if progress_tracker:
                progress_tracker.update_stage("knowledge_graph", "知识图谱构建完成", 100)
            else:
                progress_callback("knowledge_graph", "知识图谱构建完成", 100)
                
            total_time = time.time() - start_time
            logger.info(f"✅ 知识图谱构建完成，耗时 {total_time:.2f} 秒")
            return index
            
        except Exception as e:
            error_msg = f"构建知识图谱失败: {e}"
            logger.error(error_msg)
            if progress_tracker:
                progress_tracker.error("knowledge_graph", error_msg)
            else:
                progress_callback("knowledge_graph", error_msg, 0)
            return None
    
    def query_knowledge_graph(self, query: str, index: Any = None) -> str:
        """查询知识图谱"""
        try:
            logger.info(f"查询知识图谱: {query}")
            
            if index is None:
                if not self.graph_store:
                    return "错误: 图存储未初始化"
                
                # 确保LLM和Embed Model已就绪
                if not self.llm or not self.embed_model:
                     if not self.initialize():
                         return "错误: 组件初始化失败"
                
                logger.info("正在从现有存储加载索引...")
                try:
                    index = self.modules['PropertyGraphIndex'].from_existing(
                        property_graph_store=self.graph_store,
                        llm=self.llm,
                        embed_model=self.embed_model
                    )
                except Exception as e:
                    logger.error(f"加载现有索引失败: {e}")
                    return f"加载索引失败: {str(e)}"
            
            query_engine = index.as_query_engine(
                include_text=True,
                similarity_top_k=5
            )
            
            # 添加重排序逻辑
            reranker = RerankerFactory.create_reranker()
            if reranker:
                initial_k = RERANK_CONFIG.get('initial_top_k', 10)
                logger.info(f"启用重排序: initial_k={initial_k}, model={RERANK_CONFIG.get('model')}")
                
                query_engine = index.as_query_engine(
                    include_text=True,
                    similarity_top_k=initial_k,
                    node_postprocessors=[reranker]
                )
            
            response = query_engine.query(query)
            logger.info("✅ 查询完成")
            return str(response)
            
        except Exception as e:
            logger.error(f"查询失败: {e}")
            return f"查询失败: {str(e)}"

# 全局构建器实例 - 为了保持兼容性，变量名仍为 builder
builder = KnowledgeGraphManager()

# 初始化COS上传器
cos_uploader = None
try:
    oss_config = OSSConfig(OSS_CONFIG)
    cos_uploader = COSUploader(oss_config)
    logger.info("COS上传器初始化成功")
except Exception as e:
    logger.error(f"COS上传器初始化失败: {e}")