#!/usr/bin/env python3
"""
知识图谱管理器 - 核心业务逻辑
使用工厂模式重构，负责协调各个组件的工作
"""

import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
import time
from concurrent.futures import ThreadPoolExecutor

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from config import setup_logging, DOCUMENT_CONFIG, API_CONFIG, EMBEDDING_CONFIG, NEO4J_CONFIG, OSS_CONFIG, RERANK_CONFIG
from factories import LlamaModuleFactory, ModelFactory, GraphStoreFactory, ExtractorFactory, RerankerFactory
from progress_sse import ProgressTracker, progress_callback
from oss_uploader import COSUploader, OSSConfig
from ocr_parser import DeepSeekOCRParser
from enhanced_entity_extractor import StandardTermMapper
import hashlib
import json
import collections

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
            
    def get_file_hash(self, file_path: str) -> str:
        """计算文件MD5"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""
            
    def is_processed(self, file_path: str) -> bool:
        """检查文件是否已处理且未修改"""
        abs_path = str(Path(file_path).absolute())
        current_hash = self.get_file_hash(abs_path)
        if not current_hash:
            return False
            
        stored_hash = self.processed_files.get(abs_path)
        return stored_hash == current_hash
        
    def mark_processed(self, file_path: str):
        """标记文件为已处理"""
        abs_path = str(Path(file_path).absolute())
        current_hash = self.get_file_hash(abs_path)
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
        self.executor = ThreadPoolExecutor(max_workers=DOCUMENT_CONFIG.get("num_workers", 4))
        self._initialized = False
        self.processed_file_manager = ProcessedFileManager()
        self.metrics = {
            "processed_docs": 0,
            "total_docs": 0,
            "entities_count": 0,
            "relationships_count": 0
        }
        
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
                    # LlamaIndex usually puts absolute path in file_path
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
            
            # 进度范围: 30% -> 90%
            start_pct = 30
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
                
                # 插入节点
                index.insert_nodes(batch)
            
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
    
    def _get_graph_counts(self, graph_store) -> tuple:
        try:
            is_neo4j = "Neo4jPropertyGraphStore" in str(type(graph_store))
            if is_neo4j:
                with graph_store._driver.session() as session:
                    e = session.run("MATCH (n:Entity) RETURN count(n) as c").single()["c"]
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
                        result = session.run("MATCH (n:Entity) RETURN DISTINCT n.name as name")
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
                             # Out edges
                             for _, nbr, data in list(G.out_edges(s_node, data=True)):
                                 if not G.has_edge(t_node, nbr):
                                     G.add_edge(t_node, nbr, **data)
                             # In edges
                             for nbr, _, data in list(G.in_edges(s_node, data=True)):
                                 if not G.has_edge(nbr, t_node):
                                     G.add_edge(nbr, t_node, **data)
                             G.remove_node(s_node)
                             count += 1
                         except Exception as e:
                             logger.warning(f"Memory merge failed for {source}->{target}: {e}")
                logger.info(f"MemoryStore: 合并了 {count} 对实体")
            else:
                logger.warning("不支持的内存图存储结构，跳过合并")
        except Exception as e:
            logger.error(f"内存图合并失败: {e}")

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
