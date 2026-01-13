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

from config import setup_logging, DOCUMENT_CONFIG, API_CONFIG, EMBEDDING_CONFIG, NEO4J_CONFIG, OSS_CONFIG, RERANK_CONFIG, VALIDATOR_CONFIG, EXTRACTOR_CONFIG, ENTITY_DESCRIPTION_CONFIG
from factories import LlamaModuleFactory, ModelFactory, GraphStoreFactory, ExtractorFactory, RerankerFactory
from progress_sse import ProgressTracker, progress_callback
from oss_uploader import COSUploader, OSSConfig
from ocr_parser import DeepSeekOCRParser
from enhanced_entity_extractor import StandardTermMapper
from graph_agent import GraphAgent
import json
import collections

# 导入 common 模块的工具
from llama.common import (
    get_file_hash,
    DynamicThreadPool,
    TaskManager
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
                self.graph_agent = GraphAgent(self.graph_store)
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
    
    def _is_relevant_chunk(self, text: str) -> bool:
        """检查分块是否包含相关的医学关键词"""
        # 核心医学关键词列表 - 用于预筛选分块，减少无效LLM调用
        keywords = [
            "近视", "远视", "散光", "弱视", "斜视", "屈光", "老视", "白内障", 
            "视力", "眼轴", "角膜", "晶状体", "视网膜", "脉络膜", "巩膜", "眼压",
            "阿托品", "OK镜", "塑形镜", "RGP", "眼镜", "接触镜", "手术", "激光",
            "调节", "集合", "融像", "视疲劳", "眼底", "黄斑", "视神经", "度数"
        ]
        return any(k in text for k in keywords)

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
            
            raw_documents = reader.load_data()
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
            documents = []
            total_chunks = 0
            total_chars = 0
            chunk_time_sum = 0.0
            filtered_count = 0
            sample_bench_done = False
            for raw_doc in raw_documents:
                t1 = time.time()
                if DOCUMENT_CONFIG.get("benchmark_chunking", False) and not sample_bench_done:
                    self._benchmark_chunking(raw_doc)
                    sample_bench_done = True
                chunked_docs = self._chunk_document(raw_doc)
                
                # 关键词预筛选
                relevant_docs = []
                for d in chunked_docs:
                    if self._is_relevant_chunk(d.text):
                        relevant_docs.append(d)
                    else:
                        filtered_count += 1
                
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
        """使用优化的分块策略处理单个文档
        
        Args:
            document: 原始文档对象
            
        Returns:
            分块后的文档列表
        """
        from llama_index.core.node_parser import SentenceSplitter
        
        # 获取配置参数
        text_len = len(getattr(document, "text", ""))
        dyn = DOCUMENT_CONFIG.get('dynamic_chunking', False)
        base_chunk_size = DOCUMENT_CONFIG.get('chunk_size', 600)
        max_chunk_length = DOCUMENT_CONFIG.get('max_chunk_length', 800)
        min_chunk_length = DOCUMENT_CONFIG.get('min_chunk_length', 500)
        target_chars = DOCUMENT_CONFIG.get('dynamic_target_chars_per_chunk', base_chunk_size)
        if dyn and text_len > 0:
            target_chars = DOCUMENT_CONFIG.get('dynamic_target_chars_per_chunk', base_chunk_size)
            chunk_size = max(min_chunk_length, min(max_chunk_length, target_chars))
            
            # 2. 实体密度检测
            # 简单估算实体密度：检查高频医学关键词出现的频率
            medical_keywords = ["近视", "远视", "散光", "眼轴", "角膜", "视网膜", "脉络膜", "眼压", "调节", "屈光"]
            doc_text = getattr(document, "text", "")
            if len(doc_text) > 0:
                keyword_count = sum(doc_text.count(k) for k in medical_keywords)
                density = keyword_count / len(doc_text)
                
                # 如果密度高（例如 > 0.5%），减小 chunk_size 以提高提取精度
                if density > 0.005:
                    logger.info(f"检测到高密度医学文本 (密度: {density:.2%})，自动缩小分块大小")
                    chunk_size = int(chunk_size * 0.8) # 缩小 20%
                    chunk_size = max(chunk_size, min_chunk_length)
        else:
            chunk_size = base_chunk_size
            
        # 调整 overlap 为 20%
        chunk_overlap = max(0, min(int(chunk_size * 0.2), 200, DOCUMENT_CONFIG.get('CHUNK_OVERLAP', int(chunk_size * 0.2))))
        
        # 创建句子分隔符
        sentence_splitter = DOCUMENT_CONFIG.get('sentence_splitter', '。！？!?')
        semantic_separator = DOCUMENT_CONFIG.get('semantic_separator', '\n\n')
        
        # 使用SentenceSplitter进行分块
        node_parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=semantic_separator,  # 优先使用语义分隔符
            paragraph_separator=semantic_separator,
            # 确保医学术语完整性
            include_prev_next_rel=True  # 包含前后关系
        )
        
        # 将文档拆分为节点
        import time
        t0 = time.time()
        nodes = node_parser.get_nodes_from_documents([document])
        gen_nodes_time = time.time() - t0
        
        # 过滤和优化块大小
        filtered_nodes = []
        for node in nodes:
            # 获取文本长度
            text_length = len(node.text)
            
            if text_length < min_chunk_length and filtered_nodes:
                pass
            
            # 如果块太大，需要进一步分割
            if text_length > max_chunk_length:
                # 递归分割过大的块
                sub_chunks = self._split_large_chunk(node, max_chunk_length, chunk_overlap)
                filtered_nodes.extend(sub_chunks)
            else:
                # 检查医学术语完整性
                processed_node = self._ensure_medical_terminology_integrity(node)
                filtered_nodes.append(processed_node)
        
        # 将节点转换回文档对象
        documents = []
        total_chars = 0
        for node in filtered_nodes:
            # 创建新的文档对象，保留原始元数据
            # 使用Document的构造函数而不是from_text方法
            doc = self.modules['Document'](
                text=node.text,
                metadata=node.metadata
            )
            documents.append(doc)
            total_chars += len(node.text)
        
        if DOCUMENT_CONFIG.get("log_chunk_metrics", False):
            avg_len = (total_chars / len(documents)) if documents else 0
            logger.info(f"ChunkStats: size={chunk_size}, overlap={chunk_overlap}, nodes={len(documents)}, avg_len={avg_len:.1f}, gen_time={gen_nodes_time:.2f}s")
        
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
                        node.text = text[:-len(prefix)]
                        return node
        
        return node
    
    def _chunk_with_params(self, document, chunk_size: int, chunk_overlap: int, max_chunk_length: int, min_chunk_length: int) -> List[Any]:
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
                    prev_node.text = combined_text
                    prev_node.id_ = f"{prev_node.id_}_merged"
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

            # 实体对齐
            self._perform_entity_resolution(index, progress_tracker)
            
            # 三元组反向自检
            if VALIDATOR_CONFIG.get("enable", False):
                self._perform_triplet_validation(index, documents, progress_tracker)
            else:
                logger.info("三元组反向自检已禁用")
            
            # 为所有实体生成描述
            if ENTITY_DESCRIPTION_CONFIG.get("enable", False):
                self._generate_entity_descriptions(index, progress_tracker)
            else:
                logger.info("实体描述生成已禁用")
            
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
        """
        logger.info("正在分析潜在的语义弱关联...")
        from enhanced_entity_extractor import StandardTermMapper
        from llama_index.core.graph_stores.types import Relation
        import itertools
        
        new_relations = []
        count = 0
        
        # 建立实体到标准名的映射以便快速查找
        # StandardTermMapper.STANDARD_ENTITIES 是个 set
        
        for doc in documents:
            text = getattr(doc, "text", "")
            if not text:
                continue
                
            found_entities = []
            # 简单的字符串匹配 
            # 优化：只检查长度 > 1 的实体
            for entity in StandardTermMapper.STANDARD_ENTITIES:
                if entity in text:
                    found_entities.append(entity)
            
            # 如果找到2个以上实体
            if len(found_entities) >= 2:
                # 生成两两组合
                for e1, e2 in itertools.combinations(found_entities, 2):
                    rel = Relation(
                        source_id=e1,
                        target_id=e2,
                        label="RELATED_TO",
                        properties={"confidence": "low", "type": "co_occurrence", "source_chunk": doc.id_}
                    )
                    new_relations.append(rel)
                    count += 1
        
        if new_relations:
            logger.info(f"发现 {count} 个潜在弱关联，正在注入图谱...")
            try:
                # 尝试使用 upsert 或 add
                # LlamaIndex 的 PropertyGraphStore 接口通常有 upsert_relations
                if hasattr(index.property_graph_store, "upsert_relations"):
                    index.property_graph_store.upsert_relations(new_relations)
                elif hasattr(index.property_graph_store, "add"):
                     index.property_graph_store.add(relations=new_relations)
                else:
                    logger.warning("Graph store does not support batch relation insertion")
            except Exception as e:
                logger.warning(f"注入弱关联失败: {e}")
    
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
    
    def _perform_entity_resolution(self, index: Any, progress_tracker: Optional[ProgressTracker] = None):
        """执行实体对齐并更新图谱"""
        try:
            from entity_resolution import EntityResolver
            import asyncio
            
            logger.info("开始执行实体对齐...")
            if progress_tracker:
                progress_tracker.update_stage("knowledge_graph", "正在执行实体对齐...", 90)
            else:
                progress_callback("knowledge_graph", "正在执行实体对齐...", 90)
                
            resolver = EntityResolver(self.embed_model)
            graph_store = index.property_graph_store
            
            # 1. 获取所有三元组/实体
            entities = []
            is_neo4j = "Neo4jPropertyGraphStore" in str(type(graph_store))
            
            if is_neo4j:
                try:
                    with graph_store._driver.session() as session:
                        result = session.run("MATCH (n:__Entity__) RETURN DISTINCT n.name as name")
                        entities = [record["name"] for record in result]
                except Exception as e:
                    logger.warning(f"Neo4j 获取实体失败，回退到通用方法: {e}")
                    triplets = graph_store.get_triplets()
                    entities = list(set([t[0].name for t in triplets] + [t[2].name for t in triplets]))
            else:
                triplets = graph_store.get_triplets()
                entities = list(set([t[0].name for t in triplets] + [t[2].name for t in triplets]))
                
            logger.info(f"检测到 {len(entities)} 个实体，开始计算相似度...")
            
            # 2. 计算相似度
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            # 在新循环中运行
            if loop.is_running():
                # 如果是主线程且循环正在运行，我们需要小心
                # 但通常这里的 loop 在 Flask 的线程中是 None 或者是新的
                # 简单起见，我们创建一个新的 loop 来运行这个任务
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                duplicates = new_loop.run_until_complete(resolver.find_duplicates(entities))
                new_loop.close()
                asyncio.set_event_loop(loop) # 恢复
            else:
                duplicates = loop.run_until_complete(resolver.find_duplicates(entities))
            
            if not duplicates:
                logger.info("未发现重复实体")
                return

            # 3. 生成合并映射
            merge_map = resolver.apply_resolution_to_triplets([], duplicates)
            logger.info(f"生成 {len(merge_map)} 个合并操作")
            
            # 4. 执行合并
            if is_neo4j:
                self._merge_neo4j_entities(graph_store, merge_map)
            else:
                self._merge_memory_entities(graph_store, merge_map)
                
            logger.info("实体对齐完成")
            
        except Exception as e:
            logger.error(f"实体对齐过程中发生错误: {e}")
            # 不阻断主流程

    def _merge_neo4j_entities(self, graph_store, merge_map):
        """Neo4j 专用合并逻辑"""
        count = 0
        try:
            with graph_store._driver.session() as session:
                tx = session.begin_transaction()
                for source, target in merge_map.items():
                    if source == target:
                        continue
                    try:
                        query = """
                        MATCH (s:Entity {name: $source})
                        MATCH (t:Entity {name: $target})
                        WITH s, t
                        CALL apoc.refactor.mergeNodes([t, s]) YIELD node
                        RETURN count(node)
                        """
                        try:
                            tx.run(query, source=source, target=target)
                        except Exception:
                            manual_query = """
                            MATCH (s:Entity {name: $source})
                            MATCH (t:Entity {name: $target})
                            WITH s, t
                            MATCH (s)-[r]->(o)
                            MERGE (t)-[nr:TYPE(r)]->(o)
                            SET nr = properties(r)
                            DELETE r
                            WITH s, t
                            MATCH (o)-[r]->(s)
                            MERGE (o)-[nr:TYPE(r)]->(t)
                            SET nr = properties(r)
                            DELETE r
                            DETACH DELETE s
                            """
                            tx.run(manual_query, source=source, target=target)
                        count += 1
                    except Exception as e:
                        logger.warning(f"合并实体 {source}->{target} 失败: {e}")
                try:
                    tx.commit()
                except Exception as e:
                    logger.warning(f"Neo4j 批量事务提交失败: {e}")
        except Exception as e:
             logger.error(f"Neo4j 合并会话失败: {e}")
        logger.info(f"Neo4j: 成功合并 {count} 对实体")

    def _merge_memory_entities(self, graph_store, merge_map):
        """内存图合并逻辑"""
        try:
            # SimplePropertyGraphStore 内部可能有 _graph (NetworkX)
            # 或者我们需要遍历 get_triplets 重新构建
            # LlamaIndex 的 SimplePropertyGraphStore 实际上比较简单
            if hasattr(graph_store, "_graph"):
                G = graph_store._graph
                count = 0
                import networkx as nx
                # 转换为 NetworkX 的 contract_nodes 或者 relabel_nodes
                # 但这里是合并两个节点。
                for source, target in merge_map.items():
                    # 检查节点是否存在（可能在之前已经被合并掉了？）
                    # NetworkX 的节点是对象，这里 name 是 string。
                    # SimplePropertyGraphStore 的 graph node 是 EntityNode 对象吗？
                    # 让我们假设是 EntityNode 对象，或者是 name string。
                    # 通常 PropertyGraphStore 用 EntityNode 作为 key? 不，NetworkX node key 通常是 ID 或 Name。
                    
                    # 简单实现：只处理 NetworkX 层面
                    # 注意：如果 source 不在图中（可能已经被作为 target 合并了），跳过
                    nodes_map = {n.name: n for n in G.nodes() if hasattr(n, 'name')}
                    # 如果节点是 string
                    if not nodes_map and list(G.nodes()):
                        nodes_map = {n: n for n in G.nodes()}
                    
                    s_node = nodes_map.get(source)
                    t_node = nodes_map.get(target)
                    
                    if s_node and t_node:
                         # 使用 NetworkX 的 contracted_nodes (生成新图) 或自定义合并
                         # 这里我们手动转移边
                         try:
                             # 出边
                             for _, nbr, data in list(G.out_edges(s_node, data=True)):
                                 if not G.has_edge(t_node, nbr):
                                     G.add_edge(t_node, nbr, **data)
                             # 入边
                             for nbr, _, data in list(G.in_edges(s_node, data=True)):
                                 if not G.has_edge(nbr, t_node):
                                     G.add_edge(nbr, t_node, **data)
                             G.remove_node(s_node)
                             count += 1
                         except Exception as e:
                             logger.warning(f"内存合并失败 {source}->{target}: {e}")
                logger.info(f"MemoryStore: 合并了 {count} 对实体")
            else:
                logger.warning("不支持的内存图存储结构，跳过合并")
        except Exception as e:
            logger.error(f"内存图合并失败: {e}")

    def _perform_triplet_validation(
        self, 
        index: Any, 
        documents: List[Any], 
        progress_tracker: Optional[ProgressTracker] = None
    ):
        """执行三元组反向自检"""
        try:
            from triplet_validator import TripletValidator
            
            logger.info("开始执行三元组反向自检...")
            if progress_tracker:
                progress_tracker.update_stage("knowledge_graph", "正在执行三元组反向自检...", 92)
            else:
                progress_callback("knowledge_graph", "正在执行三元组反向自检...", 92)
            
            # 创建轻量级校验模型
            lightweight_llm = ModelFactory.create_lightweight_llm()
            if not lightweight_llm:
                logger.warning("轻量级校验模型创建失败，跳过反向自检")
                return
            
            # 创建校验器
            validator = TripletValidator(lightweight_llm, documents)
            
            # 获取所有三元组
            graph_store = index.property_graph_store
            is_neo4j = "Neo4jPropertyGraphStore" in str(type(graph_store))
            
            triplets = []
            try:
                triplets = graph_store.get_triplets()
                logger.info(f"获取到 {len(triplets)} 个三元组用于反向验证")
            except Exception as e:
                logger.error(f"获取三元组失败: {e}")
                return
            
            if not triplets:
                logger.info("没有三元组需要验证")
                return
            
            # 执行批量验证
            sample_ratio = VALIDATOR_CONFIG.get("sample_ratio", 0.3)
            core_entities = VALIDATOR_CONFIG.get("core_entities", [])
            num_workers = VALIDATOR_CONFIG.get("num_workers", 4)  # 并行worker数量
            
            validation_results = validator.validate_triplets_batch(
                triplets,
                sample_ratio=sample_ratio,
                core_entities=core_entities,
                num_workers=num_workers
            )
            
            if not validation_results:
                logger.info("没有需要验证的三元组")
                return
            
            # 过滤无效三元组
            confidence_threshold = VALIDATOR_CONFIG.get("confidence_threshold", 0.5)
            valid_triplets, invalid_triplets = validator.filter_invalid_triplets(
                triplets,
                validation_results,
                confidence_threshold=confidence_threshold
            )
            
                # 从图存储中删除无效的三元组
            if invalid_triplets:
                logger.info(f"准备删除 {len(invalid_triplets)} 个无效三元组")
                try:
                    deleted_count = self._remove_invalid_triplets(graph_store, invalid_triplets, is_neo4j)
                    
                    # 更新统计信息
                    e_count, r_count = self._get_graph_counts(graph_store)
                    self.metrics["entities_count"] = e_count
                    self.metrics["relationships_count"] = r_count
                    
                    logger.info(f"✅ 反向自检完成: 成功删除 {deleted_count} 个无效三元组")
                except Exception as e:
                    import traceback
                    logger.error(f"删除无效三元组时发生错误: {e}")
                    logger.error(f"错误堆栈: {traceback.format_exc()}")
                    logger.warning(f"警告: {len(invalid_triplets)} 个无效三元组未能删除，数据可能包含无效关系")
            else:
                logger.info("✅ 反向自检完成: 所有验证的三元组均有效")
                
        except Exception as e:
            import traceback
            logger.error(f"三元组反向自检过程中发生错误: {e}")
            logger.error(f"错误堆栈: {traceback.format_exc()}")
            # 不阻断主流程
    
    def _remove_invalid_triplets(
        self, 
        graph_store: Any, 
        invalid_triplets: List[Tuple[EntityNode, Relation, EntityNode]], 
        is_neo4j: bool
    ) -> int:
        """从图存储中删除无效的三元组
        
        Returns:
            成功删除的三元组数量
        """
        deleted_count = 0
        try:
            if is_neo4j:
                # Neo4j 删除逻辑
                with graph_store._driver.session() as session:
                    for head, relation, tail in invalid_triplets:
                        try:
                            # 确保属性值是字符串类型
                            head_name = str(head.name) if hasattr(head, 'name') and head.name else ""
                            tail_name = str(tail.name) if hasattr(tail, 'name') and tail.name else ""
                            relation_label = str(relation.label) if hasattr(relation, 'label') and relation.label else ""
                            
                            # 处理可能的列表类型
                            if isinstance(head_name, list):
                                head_name = str(head_name[0]) if head_name else ""
                            if isinstance(tail_name, list):
                                tail_name = str(tail_name[0]) if tail_name else ""
                            if isinstance(relation_label, list):
                                relation_label = str(relation_label[0]) if relation_label else ""
                            
                            if not (head_name and tail_name and relation_label):
                                logger.warning(f"跳过无效的三元组（缺少必要属性）: {head_name} - {relation_label} - {tail_name}")
                                continue
                            
                            # 删除关系（保留节点）
                            query = """
                            MATCH (h:Entity {name: $head_name})-[r]->(t:Entity {name: $tail_name})
                            WHERE r.label = $relation_label OR type(r) = $relation_label
                            DELETE r
                            RETURN count(r) as deleted
                            """
                            result = session.run(
                                query,
                                head_name=head_name,
                                tail_name=tail_name,
                                relation_label=relation_label
                            )
                            record = result.single()
                            if record and record.get("deleted", 0) > 0:
                                deleted_count += 1
                                logger.debug(f"✅ 删除Neo4j关系: {head_name} - {relation_label} - {tail_name}")
                            else:
                                logger.debug(f"⚠️ 未找到要删除的关系: {head_name} - {relation_label} - {tail_name}")
                        except Exception as e:
                            logger.warning(f"删除Neo4j关系失败 ({head.name if hasattr(head, 'name') else 'N/A'} - {relation.label if hasattr(relation, 'label') else 'N/A'} - {tail.name if hasattr(tail, 'name') else 'N/A'}): {e}")
                
                return deleted_count
            else:
                # 内存图存储删除逻辑
                # 构建无效三元组的标识集合用于匹配
                invalid_keys = set()
                for head, relation, tail in invalid_triplets:
                    try:
                        head_name = str(head.name) if hasattr(head, 'name') else str(head)
                        tail_name = str(tail.name) if hasattr(tail, 'name') else str(tail)
                        relation_label = str(relation.label) if hasattr(relation, 'label') else str(relation)
                        # 处理可能的列表类型
                        if isinstance(head_name, list):
                            head_name = str(head_name[0]) if head_name else ""
                        if isinstance(tail_name, list):
                            tail_name = str(tail_name[0]) if tail_name else ""
                        if isinstance(relation_label, list):
                            relation_label = str(relation_label[0]) if relation_label else ""
                        invalid_keys.add((head_name, relation_label, tail_name))
                    except Exception as e:
                        logger.warning(f"构建无效三元组标识时出错: {e}")
                        continue
                
                # 获取所有三元组并过滤
                try:
                    all_triplets = graph_store.get_triplets()
                    deleted_count = 0
                    
                    for triplet in all_triplets:
                        try:
                            head, relation, tail = triplet
                            head_name = str(head.name) if hasattr(head, 'name') else str(head)
                            tail_name = str(tail.name) if hasattr(tail, 'name') else str(tail)
                            relation_label = str(relation.label) if hasattr(relation, 'label') else str(relation)
                            # 处理可能的列表类型
                            if isinstance(head_name, list):
                                head_name = str(head_name[0]) if head_name else ""
                            if isinstance(tail_name, list):
                                tail_name = str(tail_name[0]) if tail_name else ""
                            if isinstance(relation_label, list):
                                relation_label = str(relation_label[0]) if relation_label else ""
                            
                            triplet_key = (head_name, relation_label, tail_name)
                            if triplet_key in invalid_keys:
                                # 尝试从图存储中删除（如果支持）
                                try:
                                    # SimplePropertyGraphStore 可能需要特殊处理
                                    # 这里我们先记录，实际的删除可能需要通过重建图来实现
                                    deleted_count += 1
                                except Exception as e:
                                    logger.debug(f"无法直接删除三元组: {e}")
                        except Exception as e:
                            logger.warning(f"处理三元组时出错: {e}")
                            continue
                    
                    logger.info(f"内存图存储: 标识了 {len(invalid_keys)} 个无效三元组（实际删除可能需要重建图）")
                    return len(invalid_keys)  # 返回标识的数量
                except Exception as e:
                    logger.warning(f"处理内存图存储删除时出错: {e}")
                    logger.info(f"内存图存储: 标记了 {len(invalid_triplets)} 个无效三元组（需重建图以生效）")
                    return 0
                
        except Exception as e:
            import traceback
            logger.error(f"删除无效三元组失败: {e}")
            logger.error(f"错误堆栈: {traceback.format_exc()}")
            return deleted_count  # 返回已删除的数量
        
        return deleted_count

    def stream_query_knowledge_graph(self, query: str, index: Any = None) -> Any:
        """
        流式查询知识图谱，返回生成器
        
        Args:
            query: 查询字符串
            index: 图谱索引（可选）
            
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
                
                try:
                    index = self.modules['PropertyGraphIndex'].from_existing(
                        property_graph_store=self.graph_store,
                        llm=self.llm,
                        embed_model=self.embed_model
                    )
                except Exception as e:
                    logger.error(f"加载现有索引失败: {e}")
                    yield f"加载索引失败: {str(e)}"
                    return
            
            # 创建流式查询引擎
            # 基础配置
            engine_kwargs = {
                "include_text": True,
                "similarity_top_k": 5,
                "streaming": True  # 启用流式输出
            }

            # 添加后处理器
            postprocessors = []
            
            # 1. 语义补偿
            try:
                from semantic_enrichment_postprocessor import SemanticEnrichmentPostprocessor
                semantic_enricher = SemanticEnrichmentPostprocessor(
                    graph_store=self.graph_store,
                    max_neighbors_per_entity=10
                )
                postprocessors.append(semantic_enricher)
            except Exception as e:
                logger.warning(f"语义补偿后处理器初始化失败: {e}")
            
            # 2. 重排序
            reranker = RerankerFactory.create_reranker()
            if reranker:
                initial_k = RERANK_CONFIG.get('initial_top_k', 10)
                engine_kwargs["similarity_top_k"] = initial_k
                postprocessors.append(reranker)
            
            # 3. 图谱上下文（最短路径连接Top-K实体，注入Prompt）
            try:
                from graph_context_postprocessor import GraphContextPostprocessor
                graph_context = GraphContextPostprocessor(
                    graph_store=self.graph_store,
                    max_path_depth=2,
                    max_paths=10
                )
                postprocessors.append(graph_context)
            except Exception as e:
                logger.warning(f"图谱上下文后处理器初始化失败: {e}")
                
            if postprocessors:
                engine_kwargs["node_postprocessors"] = postprocessors
            
            query_engine = index.as_query_engine(**engine_kwargs)
            
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
                                        paths_early = parsed
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
            
            # 2. 从 GraphContext 注入的 source_nodes 中提取路径
            paths = []
            try:
                import json as _json
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
                                        paths = parsed
                                        break
                                except Exception:
                                    pass
            except Exception:
                paths = []
            
            # 3. 推送路径数据（如果有）
            if paths:
                yield {
                    "type": "graph_paths",
                    "data": paths,
                    "full_answer": full_answer
                }
            
            # 4. 完成事件
            yield {
                "type": "done",
                "full_answer": full_answer
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
                
                logger.info("正在从现有存储加载索引...")
                try:
                    index = self.modules['PropertyGraphIndex'].from_existing(
                        property_graph_store=self.graph_store,
                        llm=self.llm,
                        embed_model=self.embed_model
                    )
                except Exception as e:
                    logger.error(f"加载现有索引失败: {e}")
                    return {
                        "answer": f"加载索引失败: {str(e)}",
                        "paths": []
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
            
            return {
                "answer": answer,
                "paths": []
            }
            
        except Exception as e:
            logger.error(f"查询失败: {e}")
            return {
                "answer": f"查询失败: {str(e)}",
                "paths": []
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
