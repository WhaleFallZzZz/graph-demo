#!/usr/bin/env python3
"""
知识图谱管理器 - 核心业务逻辑
使用工厂模式重构，负责协调各个组件的工作
"""

import sys
import os
import re
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from llama_index.core.graph_stores.types import EntityNode, Relation

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from llama.config import setup_logging, DOCUMENT_CONFIG, API_CONFIG, EMBEDDING_CONFIG, NEO4J_CONFIG, OSS_CONFIG, RERANK_CONFIG, EXTRACTOR_CONFIG, ENTITY_DESCRIPTION_CONFIG, HYBRID_SEARCH_CONFIG
from llama.factories import LlamaModuleFactory, ModelFactory, GraphStoreFactory, ExtractorFactory, RerankerFactory
from llama.progress_sse import ProgressTracker, progress_callback
from llama.oss_uploader import COSUploader, OSSConfig
from llama.ocr_parser import DeepSeekOCRParser
# 注释 StandardTermMapper (标准词映射) 相关代码
# from enhanced_entity_extractor import StandardTermMapper
from llama.graph_agent import GraphAgent
from llama.semantic_chunker import ImprovedSemanticChunker, ImprovedSemanticSplitter
import json
import collections

# 导入 common 模块的工具
from llama.common import (
    get_file_hash,
    DynamicThreadPool
)

class DocumentIndex:
    """文档倒排索引 - 用于加速关键信息定位"""
    def __init__(self):
        self.index = collections.defaultdict(list) # keyword -> list of (doc_id, chunk_index)
        
    def build_index(self, documents: List[Any], keywords: List[str]):
        """建立关键词到文档分块的倒排索引"""
        logger.info(f"正在为 {len(documents)} 个文档分块建立倒排索引...")
        start_time = time.time()
        
        for idx, doc in enumerate(documents):
            text = getattr(doc, "text", "")
            doc_id = getattr(doc, "id_", str(idx))
            
            # 检查每个关键词
            for keyword in keywords:
                if keyword in text:
                    self.index[keyword].append((doc_id, idx))
                    
        elapsed = time.time() - start_time
        logger.info(f"倒排索引建立完成，耗时 {elapsed:.2f}s，包含 {len(self.index)} 个关键词条目")
        return self.index

# 设置日志
logger = setup_logging()

class ProcessedFileManager:
    """已处理文件管理器 - 支持增量更新"""
    def __init__(self, record_file: str = "processed_files.json"):
        self.record_file = Path(os.getcwd()) / record_file
        self.processed_files = self._load_records()
        self._dirty = False
        
    def _load_records(self) -> Dict[str, str]:
        if self.record_file.exists():
            try:
                with open(self.record_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"无法读取处理记录: {e}")
                return {}
        return {}
        
    def save_records(self):
        try:
            with open(self.record_file, 'w', encoding='utf-8') as f:
                json.dump(self.processed_files, f, ensure_ascii=False, indent=2)
            self._dirty = False
        except Exception as e:
            logger.error(f"保存处理记录失败: {e}")
            
    def is_processed(self, file_path: str) -> bool:
        """检查文件是否已处理且未修改"""
        abs_path = str(Path(file_path).absolute())
        # 使用 common 模块的 get_file_hash
        current_hash = get_file_hash(abs_path)
        if not current_hash:
            return False
            
        stored_hash = self.processed_files.get(abs_path)
        return stored_hash == current_hash
        
    def mark_processed(self, file_path: str):
        """标记文件为已处理"""
        abs_path = str(Path(file_path).absolute())
        # 使用 common 模块的 get_file_hash
        current_hash = get_file_hash(abs_path)
        if current_hash:
            self.processed_files[abs_path] = current_hash
            self._dirty = True

class KnowledgeGraphManager:
    """知识图谱管理器 - 核心Facade"""
    
    def __init__(self):
        """初始化知识图谱管理器"""
        self.modules = None
        self.llm = None
        self.embed_model = None
        self.graph_store = None
        # 使用 common 模块的 DynamicThreadPool 替代 ThreadPoolExecutor
        self.thread_pool = DynamicThreadPool(
            min_workers=2,
            max_workers=DOCUMENT_CONFIG.get("num_workers", 4),
            idle_timeout=60.0
        )
        self._initialized = False
        self.processed_file_manager = ProcessedFileManager()
        self.metrics = {
            "processed_docs": 0,
            "total_docs": 0,
            "entities_count": 0,
            "relationships_count": 0
        }
        self.graph_agent = None  # 智能图谱查询代理
        
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
            
            # 检查图存储类型
            store_type = type(self.graph_store).__name__
            logger.info(f"图存储类型: {store_type}")
            if "Neo4jPropertyGraphStore" not in store_type:
                logger.warning(f"⚠️ 警告: 当前使用的是 {store_type} 而非 Neo4jPropertyGraphStore。数据将不会持久化到 Neo4j！")
                progress_callback("initialization", f"⚠️ 警告: 未检测到Neo4j配置，数据将不会保存！", 60)
            
            # 测试连接
            progress_callback("initialization", "正在测试数据库连接...", 80)
            try:
                self.graph_store.structured_query("MATCH (n) RETURN count(n) LIMIT 1")
                logger.info("Neo4j连接测试成功")
            except Exception as e:
                logger.warning(f"Neo4j连接测试失败: {e}")
                # 不中断，继续执行，因为可能是网络波动
            
            # 5. 初始化智能图谱查询代理
            progress_callback("initialization", "正在初始化智能图谱查询代理...", 90)
            try:
                # 传入 LLM 实例以支持 LLM 意图分类器
                self.graph_agent = GraphAgent(self.graph_store, llm_instance=self.llm)
                logger.info("✅ 智能图谱查询代理初始化成功")
            except Exception as e:
                logger.warning(f"智能图谱查询代理初始化失败: {e}")
                self.graph_agent = None
            
            progress_callback("initialization", "初始化完成", 100)
            self._initialized = True
            logger.info("✅ 知识图谱管理器初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"初始化失败: {e}")
            progress_callback("initialization", f"初始化失败: {str(e)}", 0)
            return False
    
    def cleanup(self):
        """清理资源，释放内存"""
        try:
            logger.info("开始清理资源...")
            
            # 清理线程池
            if hasattr(self, 'thread_pool') and self.thread_pool:
                try:
                    self.thread_pool.shutdown(wait=True)
                    logger.info("✅ 线程池已关闭")
                except Exception as e:
                    logger.warning(f"关闭线程池失败: {e}")
            
            # 清理 LLM
            if hasattr(self, 'llm') and self.llm:
                try:
                    del self.llm
                    self.llm = None
                    logger.info("✅ LLM 已清理")
                except Exception as e:
                    logger.warning(f"清理 LLM 失败: {e}")
            
            # 清理 Embedding 模型
            if hasattr(self, 'embed_model') and self.embed_model:
                try:
                    del self.embed_model
                    self.embed_model = None
                    logger.info("✅ Embedding 模型已清理")
                except Exception as e:
                    logger.warning(f"清理 Embedding 模型失败: {e}")
            
            # 清理图存储
            if hasattr(self, 'graph_store') and self.graph_store:
                try:
                    if hasattr(self.graph_store, '_driver') and self.graph_store._driver:
                        self.graph_store._driver.close()
                    del self.graph_store
                    self.graph_store = None
                    logger.info("✅ 图存储已清理")
                except Exception as e:
                    logger.warning(f"清理图存储失败: {e}")
            
            # 清理图谱代理
            if hasattr(self, 'graph_agent') and self.graph_agent:
                try:
                    del self.graph_agent
                    self.graph_agent = None
                    logger.info("✅ 图谱代理已清理")
                except Exception as e:
                    logger.warning(f"清理图谱代理失败: {e}")
            
            # 清理模块
            if hasattr(self, 'modules') and self.modules:
                try:
                    del self.modules
                    self.modules = None
                    logger.info("✅ 模块已清理")
                except Exception as e:
                    logger.warning(f"清理模块失败: {e}")
            
            self._initialized = False
            logger.info("✅ 资源清理完成")
            
        except Exception as e:
            logger.error(f"清理资源时发生错误: {e}")

    def load_documents(self, progress_tracker: Optional[ProgressTracker] = None) -> list:
        """加载文档并使用优化的分块策略"""
        try:
            if not self.modules:
                self.initialize()
                
            if progress_tracker:
                progress_tracker.update_stage("document_loading", "正在加载文档...")
            else:
                progress_callback("document_loading", "正在加载文档...", 10)
            
            # 使用SimpleDirectoryReader加载原始文档
            import time
            t0 = time.time()
            fe = {}
            try:
                if ".pdf" in DOCUMENT_CONFIG.get('supported_extensions', ['.txt', '.docx', '.pdf']):
                    fe[".pdf"] = DeepSeekOCRParser()
            except Exception as e:
                logger.warning(f"OCR解析器不可用，已跳过PDF解析: {e}")
                fe = {}
            reader = self.modules['SimpleDirectoryReader'](
                input_dir=DOCUMENT_CONFIG['path'],
                required_exts=DOCUMENT_CONFIG.get('supported_extensions', ['.txt', '.docx', '.pdf']),
                recursive=True,
                encoding='utf-8',
                file_extractor=fe
            )
            
            try:
                raw_documents = reader.load_data()
            except Exception as e:
                logger.error(f"OCR解析或PDF解析失败，将跳过PDF并重试: {e}")
                try:
                    fallback_exts = [ext for ext in DOCUMENT_CONFIG.get('supported_extensions', ['.txt', '.docx', '.pdf']) if ext.lower() != '.pdf']
                    reader_no_pdf = self.modules['SimpleDirectoryReader'](
                        input_dir=DOCUMENT_CONFIG['path'],
                        required_exts=fallback_exts,
                        recursive=True,
                        encoding='utf-8',
                        file_extractor={}
                    )
                    raw_documents = reader_no_pdf.load_data()
                    logger.info("已跳过PDF文件，其他类型文档加载成功")
                except Exception as e2:
                    logger.error(f"降级重试仍失败: {e2}")
                    raw_documents = []
            load_time = time.time() - t0
            logger.info(f"文档加载耗时 {load_time:.2f}s, 原始文档数 {len(raw_documents)}")
            
            # 增量处理：过滤已处理的文档
            if DOCUMENT_CONFIG.get("incremental_processing", True):
                new_raw_docs = []
                for doc in raw_documents:
                    file_path = doc.metadata.get('file_path') or doc.metadata.get('file_name')
                    # 如果是绝对路径，直接使用；如果是文件名，尝试拼接（不太准确，最好是full path）
                    # LlamaIndex 通常将绝对路径放在 file_path 中
                    if file_path and self.processed_file_manager.is_processed(str(file_path)):
                        logger.debug(f"跳过已处理文档: {file_path}")
                        continue
                    new_raw_docs.append(doc)
                
                skipped_count = len(raw_documents) - len(new_raw_docs)
                if skipped_count > 0:
                    logger.info(f"增量处理: 跳过了 {skipped_count} 个未修改文档")
                raw_documents = new_raw_docs

            # 使用自定义的分块策略处理文档
            # 支持多线程处理以加速 chunk 分割
            documents = []
            total_chunks = 0
            total_chars = 0
            chunk_time_sum = 0.0
            filtered_count = 0
            sample_bench_done = False
            
            # 获取多线程配置
            use_multithreading = DOCUMENT_CONFIG.get("use_multithreading_chunking", True)
            max_workers = DOCUMENT_CONFIG.get("max_chunking_workers", 4)
            
            if use_multithreading and len(raw_documents) > 1:
                # 使用多线程并行处理文档
                logger.info(f"使用多线程处理 {len(raw_documents)} 个文档 (workers={max_workers})")
                
                def process_document(raw_doc):
                    """处理单个文档的函数，用于多线程"""
                    t1 = time.time()
                    
                    # 分块处理
                    chunked_docs = self._chunk_document(raw_doc)
                    
                    # 关键词预筛选
                    relevant_docs = []
                    doc_filtered_count = 0
                    for d in chunked_docs:
                            relevant_docs.append(d)
                    
                    chunk_time = time.time() - t1
                    doc_total_chars = sum(len(getattr(d, "text", "")) for d in relevant_docs)
                    
                    return {
                        'relevant_docs': relevant_docs,
                        'chunked_count': len(chunked_docs),
                        'filtered_count': doc_filtered_count,
                        'chunk_time': chunk_time,
                        'total_chars': doc_total_chars
                    }
                
                # 使用线程池并行处理
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # 提交所有任务
                    future_to_doc = {executor.submit(process_document, doc): doc for doc in raw_documents}
                    
                    # 收集结果
                    for future in as_completed(future_to_doc):
                        try:
                            result = future.result()
                            documents.extend(result['relevant_docs'])
                            total_chunks += result['chunked_count']
                            filtered_count += result['filtered_count']
                            chunk_time_sum += result['chunk_time']
                            total_chars += result['total_chars']
                        except Exception as e:
                            logger.error(f"处理文档时出错: {e}")
                
                logger.info(f"多线程处理完成: 总耗时 {chunk_time_sum:.2f}s, 平均每文档 {chunk_time_sum/len(raw_documents):.3f}s")
                
            else:
                # 使用单线程顺序处理文档
                logger.info(f"使用单线程处理 {len(raw_documents)} 个文档")
                
            for raw_doc in raw_documents:
                t1 = time.time()
                if DOCUMENT_CONFIG.get("benchmark_chunking", False) and not sample_bench_done:
                    self._benchmark_chunking(raw_doc)
                    sample_bench_done = True
                chunked_docs = self._chunk_document(raw_doc)
                
                # 关键词预筛选
                relevant_docs = []
                for d in chunked_docs:
                        relevant_docs.append(d)
                
                chunk_time = time.time() - t1
                chunk_time_sum += chunk_time
                documents.extend(relevant_docs)
                total_chunks += len(chunked_docs) # 记录总块数（包括被过滤的）
                for d in relevant_docs:
                    total_chars += len(getattr(d, "text", ""))
            
            if filtered_count > 0:
                logger.info(f"关键词预筛选: 过滤了 {filtered_count} 个无关分块")
            
            msg = f"成功加载 {len(documents)} 个有效文档块 (来自 {len(raw_documents)} 个原始文档)"
            if progress_tracker:
                progress_tracker.update_stage("document_loading", msg)
            else:
                progress_callback("document_loading", msg, 15)
                
            logger.info(f"✅ {msg}")
            if DOCUMENT_CONFIG.get("log_chunk_metrics", False):
                avg_chunk_chars = (total_chars / len(documents)) if documents else 0
                logger.info(f"分块统计: 总块数 {total_chunks}, 有效块数 {len(documents)}, 平均有效块长度 {avg_chunk_chars:.1f} 字符, 分块耗时合计 {chunk_time_sum:.2f}s")
            
            # 建立倒排索引
            if documents:
                try:
                    indexer = DocumentIndex()
                    # 使用预定义的医学关键词
                    keywords = [
                        "近视", "远视", "散光", "弱视", "斜视", "屈光", "老视", "白内障", 
                        "视力", "眼轴", "角膜", "晶状体", "视网膜", "脉络膜", "巩膜", "眼压",
                        "阿托品", "OK镜", "塑形镜", "RGP", "眼镜", "接触镜", "手术", "激光"
                    ]
                    self.document_index = indexer.build_index(documents, keywords)
                except Exception as e:
                    logger.warning(f"建立倒排索引失败: {e}")

            return documents
            
        except Exception as e:
            error_msg = f"加载文档失败: {e}"
            logger.error(error_msg)
            if progress_tracker:
                progress_tracker.error("document_loading", error_msg)
            else:
                progress_callback("document_loading", error_msg, 0)
            return []
    
    def _chunk_document(self, document) -> List[Any]:
        """使用改进的语义分割策略处理单个文档
        
        采用两阶段策略：
        1. 结构化切分：按段落（双换行 \n\n）切分，保留基本排版逻辑
        2. 语义聚合：计算相邻段落相似度，高相似度则合并，直到达到大小限制
        3. 重叠保留：每个 chunk 保留 10%-15% 的重复内容
        
        Args:
            document: 原始文档对象
            
        Returns:
            分块后的文档列表
        """
        # 获取配置参数
        text_len = len(getattr(document, "text", ""))
        
        # 文档分块诊断日志：记录原始文档信息
        logger.info(f"📄 文档分块诊断 - 原始文本长度: {text_len:,} 字符")
        
        use_semantic = DOCUMENT_CONFIG.get('use_semantic_chunking', True)
        dyn = DOCUMENT_CONFIG.get('dynamic_chunking', False)
        base_chunk_size = DOCUMENT_CONFIG.get('chunk_size', 1024)
        max_chunk_length = DOCUMENT_CONFIG.get('max_chunk_length', 1400)
        min_chunk_length = DOCUMENT_CONFIG.get('min_chunk_length', 600)
        target_chars = DOCUMENT_CONFIG.get('dynamic_target_chars_per_chunk', base_chunk_size)
        
        # 动态调整 chunk_size
        if dyn and text_len > 0:
            target_chars = DOCUMENT_CONFIG.get('dynamic_target_chars_per_chunk', base_chunk_size)
            chunk_size = max(min_chunk_length, min(max_chunk_length, target_chars))
            
            # 实体密度检测
            medical_keywords = ["近视", "远视", "散光", "眼轴", "角膜", "视网膜", "脉络膜", "眼压", "调节", "屈光"]
            doc_text = getattr(document, "text", "")
            if len(doc_text) > 0:
                keyword_count = sum(doc_text.count(k) for k in medical_keywords)
                density = keyword_count / len(doc_text)
                
                if density > 0.005:
                    logger.info(f"检测到高密度医学文本 (密度: {density:.2%})，自动缩小分块大小")
                    chunk_size = int(chunk_size * 0.8)
                    chunk_size = max(chunk_size, min_chunk_length)
        else:
            chunk_size = base_chunk_size
            
        # 使用改进的语义分割器
        import time
        t0 = time.time()
        
        if use_semantic:
            # 使用改进的语义分割器
            logger.debug("使用改进的语义分割器进行分块（段落切分 + 语义聚合 + 重叠保留）")
            embedding_model = self.modules.get('embedding_model')
            
            # 使用用户指定的参数
            improved_chunker = ImprovedSemanticChunker(
                embedding_model=embedding_model,
                chunk_size=chunk_size,
                overlap_ratio=0.12,  # 12% 重叠
                similarity_threshold=0.70,  # 相似度阈值
                min_chunk_length=min_chunk_length,
                max_chunk_length=max_chunk_length
            )
            
            # 直接分割文本
            doc_text = getattr(document, "text", "")
            chunks = improved_chunker.split_text(doc_text)
            
            # 将 chunks 转换为节点
            nodes = []
            for i, chunk in enumerate(chunks):
                metadata = getattr(document, "metadata", {}).copy()
                metadata["chunk_index"] = i
                metadata["chunk_total"] = len(chunks)
                metadata["chunking_method"] = "improved_semantic"
                metadata["overlap_ratio"] = 0.12
                metadata["similarity_threshold"] = 0.70
                
                node = self.modules['Document'](text=chunk, metadata=metadata)
                nodes.append(node)
        else:
            # 使用传统的句子分割器
            logger.debug("使用传统句子分割器进行分块")
            from llama_index.core.node_parser import SentenceSplitter
            chunk_overlap = max(0, min(int(chunk_size * 0.2), 200, DOCUMENT_CONFIG.get('CHUNK_OVERLAP', int(chunk_size * 0.2))))
            sentence_splitter = DOCUMENT_CONFIG.get('sentence_splitter', '。！？!?')
            semantic_separator = DOCUMENT_CONFIG.get('semantic_separator', '\n\n')
            
            node_parser = SentenceSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator=semantic_separator,
                paragraph_separator=semantic_separator,
                include_prev_next_rel=True
            )
            
            nodes = node_parser.get_nodes_from_documents([document])
        
        gen_nodes_time = time.time() - t0
        
        # 过滤和优化块大小
        filtered_nodes = []
        for node in nodes:
            text_length = len(node.text)
            
            if text_length < min_chunk_length and filtered_nodes:
                pass
            
            if text_length > max_chunk_length:
                sub_chunks = self._split_large_chunk(node, max_chunk_length, int(chunk_size * 0.12))
                filtered_nodes.extend(sub_chunks)
            else:
                processed_node = self._ensure_medical_terminology_integrity(node)
                filtered_nodes.append(processed_node)
        
        # 将节点转换回文档对象
        documents = []
        total_chars = 0
        for node in filtered_nodes:
            doc = self.modules['Document'](
                text=node.text,
                metadata=node.metadata
            )
            documents.append(doc)
            total_chars += len(node.text)
        
        # 过滤掉字数太少（<50字）或中文极少（可能是纯图乱码）的 Chunk
        import re
        filtered_documents = []
        noise_count = 0
        for doc in documents:
            text = getattr(doc, "text", "")
            text_len = len(text)
            
            # 检查1: 字数是否 >= 50
            if text_len < 50:
                noise_count += 1
                continue
            
            # 检查2: 中文字符占比（中文字符应该占一定比例，避免纯图乱码）
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
            chinese_ratio = chinese_chars / text_len if text_len > 0 else 0
            
            # 如果文本长度在 50-200 之间，要求中文占比 >= 30%
            # 如果文本长度 > 200，要求中文占比 >= 20%
            min_chinese_ratio = 0.30 if text_len <= 200 else 0.20
            if chinese_ratio < min_chinese_ratio:
                noise_count += 1
                continue
            
            filtered_documents.append(doc)
        
        # 更新 documents 和统计信息
        documents = filtered_documents
        total_chars = sum(len(getattr(d, "text", "")) for d in documents)
        
        if noise_count > 0:
            logger.info(f"🧹 过滤 OCR 噪音: 移除了 {noise_count} 个无效 chunk（字数<50 或中文占比过低）")
        
        if DOCUMENT_CONFIG.get("log_chunk_metrics", True):
            avg_len = (total_chars / len(documents)) if documents else 0
            chunker_type = "改进语义" if use_semantic else "传统"
            
            # 增强的诊断日志：包含原始文本长度、分块后数量、平均 chunk 长度
            logger.info(
                f"ChunkStats[{chunker_type}]: size={chunk_size}, overlap=12%, "
                f"原始长度={text_len:,} 字符, "
                f"分块后数量={len(documents)}, "
                f"平均 chunk 长度={avg_len:.1f} 字符, "
                f"生成时间={gen_nodes_time:.2f}s"
            )
            
            # 计算预期的 chunk 数量（用于对比验证）
            if chunk_size > 0:
                expected_chunks = (text_len - int(chunk_size * 0.12)) / (chunk_size - int(chunk_size * 0.12))
                logger.info(
                    f"📊 分块对比: 实际={len(documents)} 个 chunks, "
                    f"理论预期≈{expected_chunks:.0f} 个 chunks "
                    f"(基于 chunk_size={chunk_size}, overlap={int(chunk_size * 0.12)})"
                )
        
        return documents
    
    def _split_large_chunk(self, node, max_length: int, overlap: int) -> List[Any]:
        """递归分割过大的文本块
        
        Args:
            node: 节点对象
            max_length: 最大长度
            overlap: 重叠字符数
            
        Returns:
            分割后的节点列表
        """
        text = node.text
        if len(text) <= max_length:
            return [node]
        
        # 找到合适的分割点（优先在句子边界分割）
        split_points = []
        current_pos = 0
        
        # 查找句子分隔符
        sentence_separators = list(DOCUMENT_CONFIG.get('sentence_splitter', '。！？!?'))
        
        while current_pos < len(text) - max_length:
            # 在最大长度附近查找句子分隔符
            search_start = current_pos + max_length - 100  # 留出100字符的搜索空间
            search_end = min(current_pos + max_length, len(text))
            
            split_pos = -1
            for sep in sentence_separators:
                # 从后往前搜索，找到最接近max_length的分隔符
                pos = text.rfind(sep, current_pos, search_end)
                if pos != -1 and pos > current_pos:
                    split_pos = pos + 1  # 包含分隔符
                    break
            
            # 如果没有找到合适的句子分隔符，就在最大长度处分割
            if split_pos == -1:
                split_pos = min(current_pos + max_length, len(text))
            
            split_points.append(split_pos)
            current_pos = split_pos
        
        # 创建分割后的节点
        nodes = []
        start_pos = 0
        for end_pos in split_points:
            chunk_text = text[start_pos:end_pos]
            
            # 创建新节点
            new_node = self.modules['Document'](
                text=chunk_text,
                metadata=node.metadata.copy()
            )
            nodes.append(new_node)
            
            # 更新起始位置，考虑重叠
            start_pos = max(end_pos - overlap, 0)
        
        # 处理最后一个块
        if start_pos < len(text):
            last_chunk = text[start_pos:]
            if len(last_chunk) > 0:  # 确保不是空块
                nodes.append(self.modules['Document'](text=last_chunk, metadata=node.metadata.copy()))
        
        return nodes
    
    def _ensure_medical_terminology_integrity(self, node) -> Any:
        """确保医学术语完整性
        增加边界检测：确保每个实体的首尾都出现在同一chunk中
        """
        text = node.text
        
        # 关键医学术语列表，用于检查边界截断
        critical_terms = [
            "角膜塑形镜", "低浓度阿托品", "眼轴长度", "病理性近视", "视网膜脱落",
            "调节幅度", "LogMAR视力表", "全飞秒激光手术", "准分子激光手术"
        ]
        
        # 常见的有效子术语（如果截断在这个位置，是可以接受的，或者是独立的实体）
        valid_subterms = ["角膜", "视网膜", "近视", "调节", "眼轴", "手术", "激光"]
        
        # 检查末尾截断
        # 如果文本以某个术语的前缀结尾（但不是完整术语），且该前缀本身不是有效术语，则截断它
        # 依靠 overlap 在下一个 chunk 中完整读取
        for term in critical_terms:
            # 检查长度至少为2的前缀
            for i in range(2, len(term)):
                prefix = term[:i]
                if text.endswith(prefix):
                    # 检查是否已经是完整术语（通过是否能匹配更长的前缀来判断 - 循环会继续）
                    # 但在这里我们只看当前 prefix。如果 text 以 prefix 结尾，
                    # 我们需要确认它不是完整 term 的一部分（即 text 结尾就是 prefix，而不是 prefix + ...）
                    # text.endswith(prefix) 已经是确认了。
                    
                    # 只要长度不等于 term 的长度，就是部分匹配
                    if len(prefix) < len(term):
                        # 检查这个前缀是否本身就是有效词
                        if prefix in valid_subterms:
                            continue
                            
                        # 这是一个不完整的截断，例如 "角膜塑"
                        # 我们将其移除，让下一个 chunk (有 overlap) 来处理完整的 "角膜塑形镜"
                        logger.debug(f"边界检测: 发现末尾截断的术语片段 '{prefix}' (原词: {term})，已自动修剪")
                        # 创建新的 Document 对象来替换原来的对象
                        new_text = text[:-len(prefix)]
                        return self.modules['Document'](text=new_text, metadata=node.metadata.copy())
        
        return node
    
    def _chunk_with_params(self, document, chunk_size: int, chunk_overlap: int, max_chunk_length: int, min_chunk_length: int) -> List[Any]:
        use_semantic = DOCUMENT_CONFIG.get('use_semantic_chunking', True)
        
        if use_semantic:
            embedding_model = self.modules.get('embedding_model')
            
            semantic_splitter = ImprovedSemanticSplitter(
                embedding_model=embedding_model,
                chunk_size=chunk_size,
                overlap_ratio=chunk_overlap / chunk_size if chunk_size > 0 else 0.12,
                min_chunk_length=min_chunk_length,
                max_chunk_length=max_chunk_length,
                similarity_threshold=DOCUMENT_CONFIG.get('similarity_threshold', 0.75),
                paragraph_separator=DOCUMENT_CONFIG.get('semantic_separator', '\n\n')
            )
            
            nodes = semantic_splitter.get_nodes_from_documents([document])
        else:
            from llama_index.core.node_parser import SentenceSplitter
            sentence_splitter = DOCUMENT_CONFIG.get('sentence_splitter', '。！？!?')
            semantic_separator = DOCUMENT_CONFIG.get('semantic_separator', '\n\n')
            node_parser = SentenceSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator=semantic_separator,
                paragraph_separator=semantic_separator,
                include_prev_next_rel=True
            )
            nodes = node_parser.get_nodes_from_documents([document])
        
        filtered_nodes = []
        for node in nodes:
            text_length = len(node.text)
            if text_length < min_chunk_length and filtered_nodes:
                prev_node = filtered_nodes[-1]
                combined_text = prev_node.text + " " + node.text
                if len(combined_text) <= max_chunk_length:
                    merged_node = self.modules['Document'](text=combined_text, metadata=prev_node.metadata.copy())
                    merged_node.id_ = f"{prev_node.id_}_merged"
                    filtered_nodes[-1] = merged_node
                    continue
            if text_length > max_chunk_length:
                sub_chunks = self._split_large_chunk(node, max_chunk_length, chunk_overlap)
                filtered_nodes.extend(sub_chunks)
            else:
                filtered_nodes.append(node)
        docs = []
        for n in filtered_nodes:
            docs.append(self.modules['Document'](text=n.text, metadata=n.metadata))
        return docs
    
    def _benchmark_chunking(self, document):
        import time
        old_size = 600
        old_overlap = 80
        old_max = 800
        old_min = 500
        t0 = time.time()
        old_docs = self._chunk_with_params(document, old_size, old_overlap, old_max, old_min)
        old_t = time.time() - t0
        new_size = DOCUMENT_CONFIG.get('chunk_size', 1024)
        new_overlap = DOCUMENT_CONFIG.get('chunk_overlap', 120)
        new_max = DOCUMENT_CONFIG.get('max_chunk_length', 1400)
        new_min = DOCUMENT_CONFIG.get('min_chunk_length', 600)
        t1 = time.time()
        new_docs = self._chunk_with_params(document, new_size, new_overlap, new_max, new_min)
        new_t = time.time() - t1
        old_chars = sum(len(d.text) for d in old_docs)
        new_chars = sum(len(d.text) for d in new_docs)
        logger.info(f"BenchmarkChunking: old(size={old_size},overlap={old_overlap}) chunks={len(old_docs)} time={old_t:.2f}s avg_len={(old_chars/len(old_docs)) if old_docs else 0:.1f}")
        logger.info(f"BenchmarkChunking: new(size={new_size},overlap={new_overlap}) chunks={len(new_docs)} time={new_t:.2f}s avg_len={(new_chars/len(new_docs)) if new_docs else 0:.1f}")
    
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
            
            # # 0. 预处理：基于别名映射替换文本中的非标实体
            # if EXTRACTOR_CONFIG.get("alias_mapping"):
            #     logger.info("正在执行文本预处理：别名替换...")
            #     mapping = EXTRACTOR_CONFIG["alias_mapping"]
                
            #     # 预编译正则：按长度降序排序，确保优先匹配长词
            #     sorted_aliases = sorted(mapping.keys(), key=len, reverse=True)
            #     pattern_str = '|'.join(map(re.escape, sorted_aliases))
            #     pattern = re.compile(pattern_str)
                
            #     processed_count = 0
            #     for doc in documents:
            #         if not hasattr(doc, "text") or not doc.text:
            #             continue
                    
            #         original_text = doc.text
            #         # 使用正则一次性替换，避免递归替换问题 (如 AL->眼轴长度, 然后 眼轴->眼轴长度 => 眼轴长度长度)
            #         modified_text = pattern.sub(lambda m: mapping[m.group(0)], original_text)
                    
            #         if modified_text != original_text:
            #             if hasattr(doc, "set_content"):
            #                 doc.set_content(modified_text)
            #             else:
            #                 doc.text = modified_text
            #             processed_count += 1
                
            #     logger.info(f"别名替换完成，共修改了 {processed_count} 个文档")
            
            # 创建提取器
            if progress_tracker:
                progress_tracker.update_stage("knowledge_graph", "正在创建实体提取器...", 25)
                
            extractor = ExtractorFactory.create_extractor(self.llm)
            if not extractor:
                error_msg = "实体提取器创建失败"
                if progress_tracker:
                    progress_tracker.error("knowledge_graph", error_msg)
                return None
            
            total_docs = len(documents)
            self.metrics["total_docs"] = total_docs
            
            if progress_tracker:
                progress_tracker.update_stage("knowledge_graph", "正在初始化图谱索引...", 30)
            
            # 第一步：创建骨架（Document + Chunk）
            if progress_tracker:
                progress_tracker.update_stage("knowledge_graph", "正在创建文档和块骨架...", 32)
            self._create_document_chunk_skeleton(documents)
            
            # 初始化空索引 (仅建立管道)
            # 使用空列表初始化，PropertyGraphIndex 会设置好 store 和 extractors
            index = self.modules['PropertyGraphIndex'].from_documents(
                [],
                llm=self.llm,
                embed_model=self.embed_model,
                property_graph_store=self.graph_store,
                kg_extractors=[extractor],
                show_progress=True
            )
            
            # 收集待标记的文件路径
            file_paths_to_mark = set()
            for doc in documents:
                # 尝试获取文件路径
                fp = doc.metadata.get('file_path') or doc.metadata.get('file_name')
                if fp:
                    file_paths_to_mark.add(str(fp))
            
            # 批量并行处理文档
            import time
            start_time = time.time()
            
            logger.info(f"开始并行处理 {len(documents)} 个文档块...")
            
            # 使用批处理以支持细粒度进度更新
            total_docs = len(documents)
            batch_size = DOCUMENT_CONFIG.get("batch_size", 5)
            
            # 进度范围: 35% -> 90%
            start_pct = 35
            end_pct = 90
            
            for i in range(0, total_docs, batch_size):
                batch = documents[i:i + batch_size]
                current_batch_end = min(i + batch_size, total_docs)
                
                progress = start_pct + ((current_batch_end / total_docs) * (end_pct - start_pct))
                msg = f"正在处理文档块 {i + 1}-{current_batch_end}/{total_docs}"
                update_every = max(1, int(DOCUMENT_CONFIG.get("progress_update_every_batches", 1)))
                batch_index = i // batch_size
                should_update = (batch_index % update_every == 0) or (current_batch_end == total_docs)
                if should_update:
                    if progress_tracker:
                        progress_tracker.update_stage("knowledge_graph", msg, progress)
                    else:
                        if i % (batch_size * 2) == 0:
                            logger.info(f"{msg} ({progress:.1f}%)")
                
                # 插入节点（实体提取）
                index.insert_nodes(batch)
                
                # 每处理完一批后，清理内存
                if i % (batch_size * 5) == 0:
                    import gc
                    gc.collect()
                    logger.debug(f"已处理 {current_batch_end}/{total_docs} 个文档块，清理内存")
            
            # # 3. 后处理：创建语义弱关联
            # if progress_tracker:
            #     progress_tracker.update_stage("knowledge_graph", "正在分析语义弱关联...", 95)
            # self._create_semantic_relationships(documents, index)
            
            self.metrics["processed_docs"] = total_docs
            e_count, r_count = self._get_graph_counts(self.graph_store)
            self.metrics["entities_count"] = e_count
            self.metrics["relationships_count"] = r_count
            
            # 标记文件为已处理
            processed_count = 0
            for fp in file_paths_to_mark:
                self.processed_file_manager.mark_processed(fp)
                processed_count += 1
            
            if processed_count > 0:
                logger.info(f"已标记 {processed_count} 个文件为已处理")
            if getattr(self.processed_file_manager, "_dirty", False):
                self.processed_file_manager.save_records()

            # 实体对齐 - 已注释：使用独立的 offline_entity_alignment.py 脚本
            # self._perform_entity_resolution(index, progress_tracker)
            
            # 为所有实体生成描述
            # if ENTITY_DESCRIPTION_CONFIG.get("enable", False):
            #     self._generate_entity_descriptions(index, progress_tracker)
            # else:
            #     logger.info("实体描述生成已禁用")
            
            # 创建溯源结构: (Entity)-[:MENTIONS]->(Chunk)-[:FROM]->(Document)
            self._create_provenance_structure(documents, index, progress_tracker)

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
    
    def _create_semantic_relationships(self, documents: List[Any], index: Any):
        """
        创建语义弱关联
        若同一文本块中出现两个标准实体且未建立关系，则创建 'RELATED_TO' 弱关联
        已注释：移除 StandardTermMapper (标准词映射) 相关代码
        """
        logger.info("正在分析潜在的语义弱关联...")
        # 注释 StandardTermMapper (标准词映射) 相关代码
        # from enhanced_entity_extractor import StandardTermMapper
        # from llama_index.core.graph_stores.types import Relation
        # import itertools
        
        # new_relations = []
        # count = 0
        
        # # 建立实体到标准名的映射以便快速查找
        # # StandardTermMapper.STANDARD_ENTITIES 是个 set
        
        # for doc in documents:
        #     text = getattr(doc, "text", "")
        #     if not text:
        #         continue
        #         
        #     found_entities = []
        #     # 简单的字符串匹配 
        #     # 优化：只检查长度 > 1 的实体
        #     for entity in StandardTermMapper.STANDARD_ENTITIES:
        #         if entity in text:
        #             found_entities.append(entity)
        #     
        #     # 如果找到2个以上实体
        #     if len(found_entities) >= 2:
        #         # 生成两两组合
        #         for e1, e2 in itertools.combinations(found_entities, 2):
        #             rel = Relation(
        #                 source_id=e1,
        #                 target_id=e2,
        #                 label="RELATED_TO",
        #                 properties={"confidence": "low", "type": "co_occurrence", "source_chunk": doc.id_}
        #             )
        #             new_relations.append(rel)
        #             count += 1
        
        # if new_relations:
        #     logger.info(f"发现 {count} 个潜在弱关联，正在注入图谱...")
        #     try:
        #         # 尝试使用 upsert 或 add
        #         # LlamaIndex 的 PropertyGraphStore 接口通常有 upsert_relations
        #         if hasattr(index.property_graph_store, "upsert_relations"):
        #             index.property_graph_store.upsert_relations(new_relations)
        #         elif hasattr(index.property_graph_store, "add"):
        #              index.property_graph_store.add(relations=new_relations)
        #         else:
        #             logger.warning("Graph store does not support batch relation insertion")
        #     except Exception as e:
        #         logger.warning(f"注入弱关联失败: {e}")
    
    def _get_graph_counts(self, graph_store) -> tuple:
        try:
            is_neo4j = "Neo4jPropertyGraphStore" in str(type(graph_store))
            if is_neo4j:
                with graph_store._driver.session() as session:
                    # PropertyGraphIndex 使用 __Entity__ 标签
                    e = session.run("MATCH (n:__Entity__) RETURN count(n) as c").single()["c"]
                    r = session.run("MATCH ()-[r]->() RETURN count(r) as c").single()["c"]
                    return int(e or 0), int(r or 0)
            triplets = graph_store.get_triplets()
            entities = set()
            for t in triplets:
                entities.add(t[0].name)
                entities.add(t[2].name)
            return len(entities), len(triplets)
        except Exception:
            return 0, 0
    
    def stream_query_knowledge_graph(self, query: str, index: Any = None, hard_match_nodes: List = None, query_intent: str = None) -> Any:
        """
        流式查询知识图谱，返回生成器
        
        Args:
            query: 查询字符串
            index: 图谱索引（可选）
            hard_match_nodes: 硬匹配的节点列表（可选），来自查询前置处理
            
        Returns:
            生成器，依次生成：
            1. LLM回答的文本片段 (str)
            2. 最终的图谱路径数据 (dict)
        """
        try:
            logger.info(f"开始流式查询: {query}")
            
            if index is None:
                if not self.graph_store:
                    yield "错误: 图存储未初始化"
                    return
                
                # 确保LLM和Embed Model已就绪
                if not self.llm or not self.embed_model:
                     if not self.initialize():
                         yield "错误: 组件初始化失败"
                         return
                
                # 确保 modules 已初始化
                if not self.modules:
                    logger.error("modules 未初始化")
                    yield "错误: 模块未初始化"
                    return
                
                # 检查 modules 是否是字典类型（防止被错误地赋值为函数）
                if not isinstance(self.modules, dict):
                    logger.error(f"modules 类型错误: {type(self.modules)}, 期望 dict")
                    # 尝试重新初始化
                    self.modules = LlamaModuleFactory.get_modules()
                    if not isinstance(self.modules, dict):
                        yield f"错误: 模块初始化失败，类型: {type(self.modules)}"
                        return
                
                try:
                    index = self.modules['PropertyGraphIndex'].from_existing(
                        property_graph_store=self.graph_store,
                        llm=self.llm,
                        embed_model=self.embed_model
                    )
                except Exception as e:
                    logger.error(f"加载现有索引失败: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    yield f"加载索引失败: {str(e)}"
                    return
            
            # 创建检索器：使用纯向量检索
            # 确保 HYBRID_SEARCH_CONFIG 是字典类型
            if not isinstance(HYBRID_SEARCH_CONFIG, dict):
                logger.error(f"HYBRID_SEARCH_CONFIG 类型错误: {type(HYBRID_SEARCH_CONFIG)}")
                initial_retrieval_k = 50
            else:
                initial_retrieval_k = HYBRID_SEARCH_CONFIG.get("initial_top_k", 50)
            logger.info(f"使用纯向量检索，Top K: {initial_retrieval_k}")
            
            # 添加后处理器（按漏斗式过滤顺序）
            postprocessors = []
            
            # 0. 硬匹配节点后处理器（最优先，放在最前面）
            if hard_match_nodes:
                try:
                    from llama.hard_match_postprocessor import HardMatchPostprocessor
                    hard_match_processor = HardMatchPostprocessor(hard_match_nodes)
                    postprocessors.append(hard_match_processor)
                    logger.info(f"添加硬匹配后处理器: {len(hard_match_nodes)} 个节点")
                except Exception as e:
                    logger.warning(f"硬匹配后处理器初始化失败: {e}")
            
            # 1. 初步重排序（Rerank）：从 Top 50 筛选到 Top 10，剔除明显不相关的节点
            reranker = RerankerFactory.create_reranker()
            if reranker:
                # 创建限流重排序器（只保留 Top 10）
                try:
                    from llama.limited_rerank_postprocessor import LimitedRerankPostprocessor
                    limited_reranker = LimitedRerankPostprocessor(
                        reranker=reranker,
                        top_n=10  # 重排序后只保留 Top 10
                    )
                    postprocessors.append(limited_reranker)
                    logger.info("添加初步重排序后处理器：从 Top 50 筛选到 Top 10")
                except ImportError:
                    # 降级：直接使用重排序器
                    logger.warning("LimitedRerankPostprocessor 导入失败，直接使用重排序器")
                    postprocessors.append(reranker)
            
            # 2-3. 并行图谱后处理：语义补偿 + 图谱上下文（并行执行以减少延迟）
            try:
                from semantic_enrichment_postprocessor import SemanticEnrichmentPostprocessor
                from graph_context_postprocessor import GraphContextPostprocessor
                from llama.parallel_graph_postprocessor import ParallelGraphPostprocessor
                
                # 创建语义补偿和图谱上下文后处理器实例
                semantic_enricher = SemanticEnrichmentPostprocessor(
                    graph_store=self.graph_store,
                    max_neighbors_per_entity=10,
                    query_intent=query_intent  # 传递查询意图，用于过滤邻居关系
                )
                
                graph_context = GraphContextPostprocessor(
                    graph_store=self.graph_store,
                    max_path_depth=2,
                    max_paths=10,
                    query_intent=query_intent,  # 传递查询意图，用于元路径搜索
                    enable_community_detection=True,  # 启用社区发现
                    community_threshold=0.3  # 社区密度阈值
                )
                
                # 创建并行后处理器（使用线程池并行执行）
                parallel_processor = ParallelGraphPostprocessor(
                    semantic_enricher=semantic_enricher,
                    graph_context=graph_context,
                    max_workers=2  # 两个并行任务
                )
                postprocessors.append(parallel_processor)
                logger.info(f"✅ 添加并行图谱后处理器（语义补偿 + 图谱上下文并行执行，意图: {query_intent or 'GENERAL'}）")
            except ImportError as e:
                logger.warning(f"并行后处理器导入失败，降级到串行执行: {e}")
                # 降级：串行执行
                try:
                    from semantic_enrichment_postprocessor import SemanticEnrichmentPostprocessor
                    semantic_enricher = SemanticEnrichmentPostprocessor(
                        graph_store=self.graph_store,
                        max_neighbors_per_entity=10,
                        query_intent=query_intent
                    )
                    postprocessors.append(semantic_enricher)
                    logger.info(f"添加语义补偿后处理器（串行模式）")
                except Exception as e2:
                    logger.warning(f"语义补偿后处理器初始化失败: {e2}")
                
                try:
                    from graph_context_postprocessor import GraphContextPostprocessor
                    graph_context = GraphContextPostprocessor(
                        graph_store=self.graph_store,
                        max_path_depth=2,
                        max_paths=10,
                        query_intent=query_intent,
                        enable_community_detection=True,
                        community_threshold=0.3
                    )
                    postprocessors.append(graph_context)
                    logger.info(f"添加图谱上下文后处理器（串行模式）")
                except Exception as e3:
                    logger.warning(f"图谱上下文后处理器初始化失败: {e3}")
            
            # 创建查询引擎：使用纯向量检索
            engine_kwargs = {
                "include_text": True,
                "similarity_top_k": initial_retrieval_k,
                "streaming": True
            }
            
            if postprocessors:
                engine_kwargs["node_postprocessors"] = postprocessors
            
            query_engine = index.as_query_engine(**engine_kwargs)
            logger.info("使用默认查询引擎（纯向量检索）")
            
            # 执行查询，获取流式响应对象
            # 阶段进度：检索开始
            yield {
                "type": "progress",
                "stage": "retrieval",
                "message": "开始检索与图上下文处理"
            }
            streaming_response = query_engine.query(query)
            # 阶段进度：检索完成
            yield {
                "type": "progress",
                "stage": "retrieval",
                "message": "检索完成，开始生成回答"
            }
            
            # 优先尝试从 source_nodes 中提取路径并尽早发送
            try:
                import json as _json
                paths_early = []
                if hasattr(streaming_response, "source_nodes") and streaming_response.source_nodes:
                    for node_with_score in streaming_response.source_nodes:
                        node = getattr(node_with_score, "node", node_with_score)
                        metadata = getattr(node, "metadata", {}) or {}
                        if metadata.get("node_type") == "graph_context":
                            paths_data = metadata.get("paths_data")
                            if paths_data:
                                try:
                                    parsed = _json.loads(paths_data)
                                    if isinstance(parsed, list):
                                        # 提取格式化后的路径字符串
                                        paths_early = [p.get("path_str", p) for p in parsed]
                                        break
                                except Exception:
                                    pass
                if paths_early:
                    yield {
                        "type": "graph_paths",
                        "data": paths_early
                    }
            except Exception:
                pass
            
            # 1. 实时推送LLM生成的文本
            full_answer = ""
            for token in streaming_response.response_gen:
                full_answer += token
                yield token
            
            # 2. 从 GraphContext 注入的 source_nodes 中提取路径和原始文本上下文
            paths = []
            contexts = []
            try:
                import json as _json
                if hasattr(streaming_response, "source_nodes") and streaming_response.source_nodes:
                    for node_with_score in streaming_response.source_nodes:
                        node = getattr(node_with_score, "node", node_with_score)
                        metadata = getattr(node, "metadata", {}) or {}
                        
                        # 分离图路径数据和普通文本块内容
                        if metadata.get("node_type") == "graph_context":
                            paths_data = metadata.get("paths_data")
                            if paths_data:
                                try:
                                    parsed = _json.loads(paths_data)
                                    if isinstance(parsed, list):
                                        # 提取格式化后的路径字符串
                                        paths.extend([p.get("path_str", p) for p in parsed])
                                except Exception:
                                    pass
                        else:
                            # 提取原始文本内容作为上下文
                            content = node.get_content()
                            if content and content not in contexts:
                                contexts.append(content)
            except Exception as e:
                logger.warning(f"提取上下文或路径失败: {e}")
            
            # 3. 推送路径数据（如果有）
            if paths:
                yield {
                    "type": "graph_paths",
                    "data": paths,
                    "full_answer": full_answer
                }
            
            # 4. 推送检索到的上下文（用于评估等场景）
            if contexts:
                yield {
                    "type": "retrieved_contexts",
                    "data": contexts
                }
            
            # 5. 完成事件
            yield {
                "type": "done",
                "full_answer": full_answer,
                "contexts": contexts
            }
            
        except Exception as e:
            logger.error(f"流式查询失败: {e}")
            yield f"查询出错: {str(e)}"

    def query_knowledge_graph(self, query: str, index: Any = None, return_paths: bool = True) -> Dict[str, Any]:
        """
        查询知识图谱，返回答案和图谱推理路径
        
        Args:
            query: 查询字符串
            index: 图谱索引（可选）
            return_paths: 是否返回图谱路径
            
        Returns:
            包含答案和图谱路径的字典
        """
        try:
            logger.info(f"查询知识图谱: {query}")
            
            if index is None:
                if not self.graph_store:
                    return {
                        "answer": "错误: 图存储未初始化",
                        "paths": []
                    }
                
                # 确保LLM和Embed Model已就绪
                if not self.llm or not self.embed_model:
                     if not self.initialize():
                         return {
                             "answer": "错误: 组件初始化失败",
                             "paths": []
                         }
                
                # 确保 modules 已初始化且类型正确
                if not self.modules or not isinstance(self.modules, dict):
                    logger.warning(f"modules 类型异常: {type(self.modules)}，尝试重新获取")
                    self.modules = LlamaModuleFactory.get_modules()
                    if not isinstance(self.modules, dict):
                         return {
                             "answer": "错误: 模块初始化失败",
                             "paths": [],
                             "contexts": []
                         }
                
                try:
                    index = self.modules['PropertyGraphIndex'].from_existing(
                        property_graph_store=self.graph_store,
                        llm=self.llm,
                        embed_model=self.embed_model
                    )
                except Exception as e:
                    logger.error(f"加载现有索引失败: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    return {
                        "answer": f"加载索引失败: {str(e)}",
                        "paths": [],
                        "contexts": []
                    }
            
            query_engine = index.as_query_engine(
                include_text=True,
                similarity_top_k=5
            )
            
            # 添加后处理器列表
            postprocessors = []
            initial_k = 5  # 默认值
            
            # 添加语义补偿后处理器（一度关联节点拉取）
            try:
                from semantic_enrichment_postprocessor import SemanticEnrichmentPostprocessor
                semantic_enricher = SemanticEnrichmentPostprocessor(
                    graph_store=self.graph_store,
                    max_neighbors_per_entity=10
                )
                postprocessors.append(semantic_enricher)
                logger.info("✅ 启用语义补偿后处理器（一度关联节点拉取）")
            except Exception as e:
                logger.warning(f"语义补偿后处理器初始化失败: {e}")
            
            # 添加重排序逻辑
            reranker = RerankerFactory.create_reranker()
            if reranker:
                initial_k = RERANK_CONFIG.get('initial_top_k', 10)
                logger.info(f"启用重排序: initial_k={initial_k}, model={RERANK_CONFIG.get('model')}")
                postprocessors.append(reranker)
            
            # 添加图谱上下文后处理器（在Top-K实体间建立最短路径连接，并转为自然语言注入Prompt）
            try:
                from graph_context_postprocessor import GraphContextPostprocessor
                graph_context = GraphContextPostprocessor(
                    graph_store=self.graph_store,
                    max_path_depth=2,
                    max_paths=10
                )
                postprocessors.append(graph_context)
                logger.info("✅ 启用图谱上下文后处理器（最短路径连接Top-K实体）")
            except Exception as e:
                logger.warning(f"图谱上下文后处理器初始化失败: {e}")
                
            # 如果有后处理器，应用到查询引擎
            if postprocessors:
                query_engine = index.as_query_engine(
                    include_text=True,
                    similarity_top_k=initial_k,
                    node_postprocessors=postprocessors
                )
            
            # 执行查询
            response = query_engine.query(query)
            answer = str(response)
            
            # 提取路径和上下文
            paths = []
            contexts = []
            try:
                import json as _json
                if hasattr(response, "source_nodes") and response.source_nodes:
                    for node_with_score in response.source_nodes:
                        node = getattr(node_with_score, "node", node_with_score)
                        metadata = getattr(node, "metadata", {}) or {}
                        
                        if metadata.get("node_type") == "graph_context":
                            paths_data = metadata.get("paths_data")
                            if paths_data:
                                try:
                                    parsed = _json.loads(paths_data)
                                    if isinstance(parsed, list):
                                        paths.extend([p.get("path_str", p) for p in parsed])
                                except Exception:
                                    pass
                        else:
                            content = node.get_content()
                            if content and content not in contexts:
                                contexts.append(content)
            except Exception as e:
                logger.warning(f"提取上下文或路径失败: {e}")
            
            return {
                "answer": answer,
                "paths": paths,
                "contexts": contexts
            }
            
        except Exception as e:
            logger.error(f"查询失败: {e}")
            return {
                "answer": f"查询失败: {str(e)}",
                "paths": [],
                "contexts": []
            }
    
    def generate_embeddings_for_nodes(self, node_ids: List[str] = None, node_names: List[str] = None) -> Dict[str, Any]:
        """
        为指定节点生成 embedding 向量
        
        Args:
            node_ids: 节点ID列表（Neo4j elementId）
            node_names: 节点名称列表
            
        Returns:
            包含成功和失败信息的字典
        """
        if not self.graph_store or not self.embed_model:
            return {
                "success": False,
                "message": "图存储或嵌入模型未初始化",
                "processed": 0,
                "failed": 0
            }
        
        is_neo4j = "Neo4jPropertyGraphStore" in str(type(self.graph_store))
        if not is_neo4j:
            return {
                "success": False,
                "message": "当前图存储不是 Neo4j，无法生成 embedding",
                "processed": 0,
                "failed": 0
            }
        
        try:
            processed_count = 0
            failed_count = 0
            failed_nodes = []
            
            with self.graph_store._driver.session() as session:
                # 查询需要生成 embedding 的节点
                # 条件：没有 embedding 或 source 为 manual/手工录入（手动新增的节点）
                # 如果明确指定了 node_ids，则只检查是否有 embedding，不限制 source
                if node_ids:
                    # 根据节点ID查询（明确指定ID时，只检查是否缺少embedding）
                    query = """
                    MATCH (n:__Entity__)
                    WHERE elementId(n) IN $node_ids
                    AND n.embedding IS NULL
                    AND (n.source IS NULL OR n.source IN ['manual', '手工录入'])
                    RETURN elementId(n) as id, n.name as name, COALESCE(n.label, n.type, '__Entity__') as label
                    """
                    result = session.run(query, node_ids=node_ids)
                elif node_names:
                    # 根据节点名称查询（只检查是否缺少embedding）
                    query = """
                    MATCH (n:__Entity__)
                    WHERE n.name IN $node_names
                    AND n.embedding IS NULL
                    AND (n.source IS NULL OR n.source IN ['manual', '手工录入'])
                    RETURN elementId(n) as id, n.name as name, COALESCE(n.label, n.type, '__Entity__') as label
                    """
                    result = session.run(query, node_names=node_names)
                else:
                    # 查询所有没有 embedding 的 manual/手工录入节点
                    query = """
                    MATCH (n:__Entity__)
                    WHERE n.embedding IS NULL 
                    AND (n.source IS NULL OR n.source IN ['manual', '手工录入'])
                    RETURN elementId(n) as id, n.name as name, COALESCE(n.label, n.type, '__Entity__') as label
                    LIMIT 100
                    """
                    result = session.run(query)
                
                nodes_to_process = []
                for record in result:
                    nodes_to_process.append({
                        "id": record["id"],
                        "name": record["name"],
                        "label": record["label"]  # COALESCE 后的 label，用于生成 embedding 文本
                    })
                
                if not nodes_to_process:
                    return {
                        "success": True,
                        "message": "没有需要生成 embedding 的节点",
                        "processed": 0,
                        "failed": 0
                    }
                
                logger.info(f"准备为 {len(nodes_to_process)} 个节点生成 embedding")
                
                # 批量生成 embedding
                for node_info in nodes_to_process:
                    try:
                        # 构建用于生成 embedding 的文本
                        # 格式：节点名称 + 节点类型（如果有且不是默认的Entity）
                        embed_text = node_info["name"]
                        if node_info["label"] and node_info["label"] != "Entity":
                            embed_text = f"{node_info['name']} {node_info['label']}"
                        
                        # 生成 embedding
                        logger.info(f"正在为节点 '{node_info['name']}' (ID: {node_info['id']}) 生成 embedding，文本: {embed_text}")
                        embedding = self.embed_model.get_text_embedding(embed_text)
                        
                        if not embedding or len(embedding) == 0:
                            raise ValueError(f"生成的 embedding 为空")
                        
                        logger.info(f"生成的 embedding 维度: {len(embedding)}")
                        
                        # 更新节点属性和标签（labels）
                        # 为手动新增的节点添加 'manual' 标签（Neo4j 的 labels，不是属性）
                        # 同时确保节点的 label 属性存在（如果没有则设置为 '__Entity__'）
                        update_query = """
                        MATCH (n:__Entity__)
                        WHERE elementId(n) = $node_id
                        SET n.embedding = $embedding,
                            n.updated_at = timestamp(),
                            n:manual,
                            n.label = CASE 
                                WHEN n.label IS NULL THEN '__Entity__'
                                ELSE n.label
                            END
                        RETURN n.name as name, 
                               n.embedding IS NOT NULL as has_embedding,
                               labels(n) as labels,
                               n.label as label
                        """
                        result = session.run(update_query, node_id=node_info["id"], embedding=embedding)
                        
                        # 验证是否成功更新
                        record = result.single()
                        if not record:
                            raise ValueError(f"节点 {node_info['id']} 不存在或更新失败")
                        
                        # 验证 embedding、labels 和 label 是否真的写入
                        verify_query = """
                        MATCH (n)
                        WHERE elementId(n) = $node_id
                        RETURN n.embedding IS NOT NULL as has_embedding, 
                               size(n.embedding) as embedding_size,
                               labels(n) as labels,
                               n.label as label
                        """
                        verify_result = session.run(verify_query, node_id=node_info["id"])
                        verify_record = verify_result.single()
                        
                        if verify_record and verify_record["has_embedding"]:
                            labels_info = f"，labels: {verify_record.get('labels', [])}"
                            label_info = f"，label属性: {verify_record.get('label', 'N/A')}"
                            logger.info(
                                f"✅ 已为节点 '{node_info['name']}' 生成并写入 embedding "
                                f"(维度: {verify_record.get('embedding_size', 'N/A')}{labels_info}{label_info})"
                            )
                            processed_count += 1
                        else:
                            raise ValueError(f"节点更新成功但验证时未找到 embedding 属性")
                        
                        
                    except Exception as e:
                        failed_count += 1
                        failed_nodes.append({
                            "name": node_info.get("name", "Unknown"),
                            "error": str(e)
                        })
                        logger.warning(f"❌ 为节点 '{node_info.get('name')}' 生成 embedding 失败: {e}")
                
                message = f"成功为 {processed_count} 个节点生成 embedding"
                if failed_count > 0:
                    message += f"，{failed_count} 个节点失败"
                
                return {
                    "success": True,
                    "message": message,
                    "processed": processed_count,
                    "failed": failed_count,
                    "failed_nodes": failed_nodes if failed_nodes else None
                }
                
        except Exception as e:
            logger.error(f"生成节点 embedding 时发生错误: {e}")
            return {
                "success": False,
                "message": f"生成 embedding 失败: {str(e)}",
                "processed": processed_count,
                "failed": failed_count
            }
    
    def _generate_entity_descriptions(self, index: Any, progress_tracker: Optional[ProgressTracker] = None):
        """为所有实体节点生成并更新 description 字段（多线程版本）"""
        try:
            if not self.graph_store or not self.llm:
                logger.warning("图存储或LLM未初始化，跳过实体描述生成")
                return
            
            is_neo4j = "Neo4jPropertyGraphStore" in str(type(self.graph_store))
            if not is_neo4j:
                logger.warning("当前图存储不是 Neo4j，跳过实体描述生成")
                return
            
            description_prompt_template = ENTITY_DESCRIPTION_CONFIG.get("description_prompt", "")
            num_workers = ENTITY_DESCRIPTION_CONFIG.get("num_workers", 2)
            request_delay = ENTITY_DESCRIPTION_CONFIG.get("request_delay", 0.3)
            max_retries = ENTITY_DESCRIPTION_CONFIG.get("max_retries", 3)
            retry_delay = ENTITY_DESCRIPTION_CONFIG.get("retry_delay", 5.0)
            
            if progress_tracker:
                progress_tracker.update_stage("knowledge_graph", "正在为所有实体生成描述...", 98)
            else:
                progress_callback("knowledge_graph", "正在为所有实体生成描述...", 98)
            
            logger.info("开始为所有实体节点生成描述...")
            
            # 查询所有需要处理的实体
            entities_to_process = []
            with self.graph_store._driver.session() as session:
                query = """
                MATCH (n:__Entity__)
                WHERE n.description IS NULL OR n.description = ''
                RETURN DISTINCT n.name as name, COALESCE(n.type, n.label, 'Entity') as entity_type
                LIMIT 100
                """
                result = session.run(query)
                
                for record in result:
                    entities_to_process.append({
                        "name": record["name"],
                        "type": record["entity_type"]
                    })
            
            if not entities_to_process:
                logger.info("没有需要生成描述的实体节点")
                return
            
            logger.info(f"准备为 {len(entities_to_process)} 个实体节点生成描述（使用 {num_workers} 个worker）")
            
            # 线程安全的计数器
            processed_count = 0
            failed_count = 0
            count_lock = threading.Lock()
            
            def generate_single_description(entity_info: Dict[str, str]) -> Tuple[bool, str]:
                """为单个实体生成描述（线程函数，带重试机制）"""
                nonlocal processed_count, failed_count
                
                entity_name = entity_info["name"]
                entity_type = entity_info["type"]
                
                # 构建 prompt
                prompt = description_prompt_template.format(
                    entity_name=entity_name,
                    entity_type=entity_type
                )
                
                # 带重试机制的API调用
                last_exception = None
                for attempt in range(max_retries + 1):
                    try:
                        # 请求前延迟（限流控制）
                        if attempt > 0:
                            # 重试时使用指数退避
                            wait_time = retry_delay * (2 ** (attempt - 1))
                            error_str = str(last_exception) if last_exception else ""
                            # 429错误需要更长的等待时间
                            if "429" in error_str or "Too Many Requests" in error_str or "RateLimitError" in error_str:
                                wait_time = wait_time * 2  # 429错误加倍等待
                            logger.warning(f"实体描述生成重试 ({entity_name}): 等待 {wait_time:.2f} 秒后重试 (第 {attempt + 1}/{max_retries + 1} 次)")
                            time.sleep(wait_time)
                        else:
                            # 首次请求前短暂延迟
                            time.sleep(request_delay)
                        
                        # 调用 LLM 生成描述
                        logger.debug(f"正在为实体 '{entity_name}' ({entity_type}) 生成描述...")
                        response = self.llm.complete(prompt)
                        description = response.text.strip()
                        
                        # 清理描述（移除可能的引号、多余空白等）
                        description = description.strip('"\'').strip()
                        if len(description) > 200:
                            description = description[:200] + "..."
                        
                        if not description:
                            logger.warning(f"实体 '{entity_name}' 的描述生成为空")
                            with count_lock:
                                failed_count += 1
                            return False, entity_name
                        
                        # 更新节点 description 属性（每个线程使用自己的 session）
                        with self.graph_store._driver.session() as session:
                            update_query = """
                            MATCH (n:__Entity__ {name: $entity_name})
                            SET n.description = $description,
                                n.updated_at = timestamp()
                            RETURN n.name as name
                            """
                            session.run(update_query, 
                                      entity_name=entity_name,
                                      description=description)
                        
                        logger.info(f"✅ 已为实体 '{entity_name}' 生成描述: {description[:50]}...")
                        with count_lock:
                            processed_count += 1
                        return True, entity_name
                        
                    except Exception as e:
                        last_exception = e
                        error_str = str(e)
                        
                        # 检查是否是429限流错误
                        if "429" in error_str or "Too Many Requests" in error_str or "RateLimitError" in error_str:
                            if attempt < max_retries:
                                # 429错误需要更长的等待时间，继续重试
                                continue
                            else:
                                logger.error(f"❌ 为实体 '{entity_name}' 生成描述失败（429限流，已达到最大重试次数）: {e}")
                                with count_lock:
                                    failed_count += 1
                                return False, entity_name
                        else:
                            # 其他错误，如果是最后一次尝试，记录并返回失败
                            if attempt >= max_retries:
                                logger.warning(f"❌ 为实体 '{entity_name}' 生成描述失败（已达到最大重试次数）: {e}")
                                with count_lock:
                                    failed_count += 1
                                return False, entity_name
                            # 其他错误也继续重试
                            continue
                
                # 如果所有重试都失败
                logger.error(f"❌ 为实体 '{entity_name}' 生成描述失败（所有重试均失败）: {last_exception}")
                with count_lock:
                    failed_count += 1
                return False, entity_name
            
            # 使用线程池并行处理
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # 提交所有任务
                futures = {
                    executor.submit(generate_single_description, entity_info): entity_info
                    for entity_info in entities_to_process
                }
                
                # 等待所有任务完成
                for future in as_completed(futures):
                    entity_info = futures[future]
                    try:
                        success, entity_name = future.result()
                    except Exception as exc:
                        logger.error(f"实体 '{entity_info.get('name')}' 处理时发生异常: {exc}")
                        with count_lock:
                            failed_count += 1
            
            logger.info(f"✅ 实体描述生成完成: 成功 {processed_count} 个，失败 {failed_count} 个")
                
        except Exception as e:
            logger.error(f"生成实体描述时发生错误: {e}")
            import traceback
            logger.error(f"错误堆栈: {traceback.format_exc()}")
    
    def _create_document_chunk_skeleton(self, documents: List[Any]):
        """
        第一步：创建骨架 (Document + Chunk)
        三层拓扑架构的物理层和上下文层
        
        Args:
            documents: 文档列表（已经是分块后的文档）
        """
        try:
            is_neo4j = "Neo4jPropertyGraphStore" in str(type(self.graph_store))
            if not is_neo4j:
                logger.warning("当前图存储不是 Neo4j，跳过骨架创建")
                return
            
            logger.info("开始创建文档和块骨架 (Document + Chunk)...")
            
            with self.graph_store._driver.session() as session:
                # 1. 按文档分组（根据 file_path 或 source_file_name）
                # 优化：使用生成器减少内存占用
                doc_groups = {}
                doc_chunks = {}  # doc_id -> list of (chunk_id, chunk_index, chunk_data)
                chunk_to_doc = {}  # chunk_id -> document_id
                
                for idx, doc in enumerate(documents):
                    # 获取文档的唯一标识（优先使用 file_path）
                    doc_path = doc.metadata.get('file_path') or doc.metadata.get('file_name')
                    if not doc_path:
                        doc_path = getattr(doc, 'id_', str(id(doc)))
                    
                    # 计算文件哈希作为唯一标识
                    try:
                        if os.path.exists(doc_path):
                            file_hash = get_file_hash(doc_path)
                        else:
                            # 如果文件不存在，使用路径的哈希
                            file_hash = str(hash(doc_path) % 1000000000)
                    except Exception:
                        file_hash = str(hash(doc_path) % 1000000000)
                    
                    if file_hash not in doc_groups:
                        doc_groups[file_hash] = {
                            'file_hash': file_hash,
                            'file_path': doc_path,
                            'metadata': doc.metadata
                        }
                        doc_chunks[file_hash] = []
                    
                    # 记录 chunk 信息
                    chunk_id = getattr(doc, 'id_', str(id(doc)))
                    chunk_text = getattr(doc, 'text', '')
                    # chunk_index 将在后面按文档分组后重新计算
                    page_number = doc.metadata.get('page_label') or doc.metadata.get('page_number') or 0
                    
                    doc_chunks[file_hash].append({
                        'chunk_id': chunk_id,
                        'text': chunk_text,
                        'page_number': page_number
                    })
                    chunk_to_doc[chunk_id] = file_hash
                    
                    # 定期清理内存
                    if idx % 1000 == 0:
                        import gc
                        gc.collect()
                
                # 为每个文档的 chunks 分配正确的 chunk_index
                for file_hash, chunks in doc_chunks.items():
                    for chunk_idx, chunk_info in enumerate(chunks):
                        chunk_info['chunk_index'] = chunk_idx
                
                # 2. 创建 Document 节点（物理层）
                created_docs = 0
                for file_hash, doc_info in doc_groups.items():
                    doc_metadata = doc_info['metadata']
                    file_path = doc_info['file_path']
                    
                    # 从 metadata 获取文档信息
                    file_name = doc_metadata.get('file_name') or os.path.basename(file_path)
                    upload_date = doc_metadata.get('created_at', int(time.time()))
                    url = doc_metadata.get('file_url', '')
                    
                    query = """
                    MERGE (d:Document {file_hash: $file_hash})
                    ON CREATE SET d.file_name = $file_name,
                                  d.upload_date = $upload_date,
                                  d.url = $url,
                                  d.created_at = timestamp()
                    ON MATCH SET d.updated_at = timestamp()
                    RETURN d.file_hash as file_hash
                    """
                    
                    try:
                        result = session.run(query, 
                                            file_hash=file_hash,
                                            file_name=file_name,
                                            upload_date=upload_date,
                                            url=url)
                        result.single()
                        created_docs += 1
                    except Exception as e:
                        logger.warning(f"创建 Document 节点失败 ({file_hash}): {e}")
                
                logger.info(f"✅ 创建了 {created_docs} 个 Document 节点")
                
                # 清理不再需要的 doc_groups
                del doc_groups
                import gc
                gc.collect()
                
                # 3. 创建 Chunk 节点并建立 Chunk-[:PART_OF]->Document 和 Chunk-[:NEXT]->Chunk 关系（上下文层）
                created_chunks = 0
                created_part_of_rels = 0
                created_next_rels = 0
                
                for file_hash, chunks in doc_chunks.items():
                    # 按 chunk_index 排序
                    chunks.sort(key=lambda x: x['chunk_index'])
                    
                    prev_chunk_id = None
                    for chunk_info in chunks:
                        chunk_id = chunk_info['chunk_id']
                        chunk_text = chunk_info['text']
                        chunk_index = chunk_info['chunk_index']
                        page_number = chunk_info['page_number']
                        
                        # 创建 Chunk 节点并建立 PART_OF 关系
                        query = """
                        MATCH (doc:Document {file_hash: $file_hash})
                        MERGE (c:Chunk {id: $chunk_id})
                        ON CREATE SET c.text = $text,
                                      c.chunk_index = $chunk_index,
                                      c.page_number = $page_number,
                                      c.created_at = timestamp()
                        MERGE (c)-[:PART_OF]->(doc)
                        RETURN c.id as id
                        """
                        
                        try:
                            result = session.run(query,
                                                file_hash=file_hash,
                                                chunk_id=chunk_id,
                                                text=chunk_text,
                                                chunk_index=chunk_index,
                                                page_number=page_number)
                            result.single()
                            created_chunks += 1
                            created_part_of_rels += 1
                            
                            # 建立 NEXT 关系（链式结构）
                            if prev_chunk_id:
                                next_query = """
                                MATCH (c_prev:Chunk {id: $prev_chunk_id})
                                MATCH (c:Chunk {id: $chunk_id})
                                MERGE (c_prev)-[:NEXT]->(c)
                                RETURN count(*) as count
                                """
                                try:
                                    session.run(next_query,
                                               prev_chunk_id=prev_chunk_id,
                                               chunk_id=chunk_id)
                                    created_next_rels += 1
                                except Exception as e:
                                    logger.debug(f"创建 NEXT 关系失败 ({prev_chunk_id} -> {chunk_id}): {e}")
                            
                            prev_chunk_id = chunk_id
                        except Exception as e:
                            logger.warning(f"创建 Chunk 节点失败 ({chunk_id}): {e}")
                
                logger.info(f"✅ 创建了 {created_chunks} 个 Chunk 节点，{created_part_of_rels} 个 PART_OF 关系，{created_next_rels} 个 NEXT 关系")
                logger.info(f"✅ 骨架创建完成: {created_docs} 个 Document, {created_chunks} 个 Chunk")
                
                # 清理不再需要的字典
                del doc_chunks
                del chunk_to_doc
                gc.collect()
                
        except Exception as e:
            logger.error(f"创建文档和块骨架时发生错误: {e}")
            import traceback
            logger.error(f"错误堆栈: {traceback.format_exc()}")
    
    def _create_provenance_structure(self, documents: List[Any], index: Any, progress_tracker: Optional[ProgressTracker] = None):
        """
        第二步：创建溯源关系 (Chunk)-[:MENTIONS]->(Entity)
        三层拓扑架构的语义层
        
        Args:
            documents: 文档列表（已经是分块后的文档）
            index: PropertyGraphIndex 实例
            progress_tracker: 进度跟踪器
        """
        try:
            is_neo4j = "Neo4jPropertyGraphStore" in str(type(self.graph_store))
            if not is_neo4j:
                logger.warning("当前图存储不是 Neo4j，跳过溯源关系创建")
                return
            
            if progress_tracker:
                progress_tracker.update_stage("knowledge_graph", "正在创建溯源关系...", 99)
            else:
                progress_callback("knowledge_graph", "正在创建溯源关系...", 99)
            
            logger.info("开始创建溯源关系 (Chunk)-[:MENTIONS]->(Entity)...")
            
            # 建立 chunk_id 到 chunk_text 的映射
            chunk_text_map = {}
            for doc in documents:
                chunk_id = getattr(doc, 'id_', str(id(doc)))
                chunk_text = getattr(doc, 'text', '')
                chunk_text_map[chunk_id] = chunk_text
            
            with self.graph_store._driver.session() as session:
                # 查询所有实体
                entity_query = """
                MATCH (e:__Entity__)
                RETURN DISTINCT e.name as entity_name
                LIMIT 1000
                """
                entities = []
                for record in session.run(entity_query):
                    entities.append(record["entity_name"])
                
                logger.info(f"找到 {len(entities)} 个实体，开始建立 MENTIONS 关系...")
                
                # 对于每个实体，查找包含该实体的 chunk，并建立 MENTIONS 关系（方向：Chunk -> Entity）
                created_mentions = 0
                batch_size = 100
                
                for i in range(0, len(entities), batch_size):
                    entity_batch = entities[i:i + batch_size]
                    
                    for entity_name in entity_batch:
                        # 查找包含该实体名称的 chunk，建立 (Chunk)-[:MENTIONS]->(Entity) 关系
                        mentions_query = """
                        MATCH (e:__Entity__ {name: $entity_name})
                        MATCH (c:Chunk)
                        WHERE c.text CONTAINS $entity_name
                        MERGE (c)-[:MENTIONS]->(e)
                        RETURN count(*) as count
                        """
                        
                        try:
                            result = session.run(mentions_query, entity_name=entity_name)
                            count = result.single()["count"]
                            created_mentions += count
                        except Exception as e:
                            logger.debug(f"建立 MENTIONS 关系失败 ({entity_name}): {e}")
                
                logger.info(f"✅ 创建了 {created_mentions} 个 MENTIONS 关系")
                logger.info(f"✅ 溯源关系创建完成: {created_mentions} 个 (Chunk)-[:MENTIONS]->(Entity) 关系")
                
        except Exception as e:
            logger.error(f"创建溯源关系时发生错误: {e}")
            import traceback
            logger.error(f"错误堆栈: {traceback.format_exc()}")

# 全局构建器实例 - 为了保持兼容性，变量名仍为 builder
builder = KnowledgeGraphManager()
