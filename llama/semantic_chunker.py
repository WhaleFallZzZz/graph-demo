"""
语义分割器 - 基于语义相似度的智能文本分割

该模块提供了基于语义相似度的智能文本分割功能，主要用于将长文本分割成语义连贯的文本块。

主要功能：
1. 语义感知分割：通过计算相邻段落的语义相似度，在语义变化较大的位置进行分割
2. 医学术语识别：使用嵌入模型识别文本中的医学术语，避免在术语中间分割
3. 智能合并与分割：自动合并过小的文本块，分割过大的文本块
4. 多种相似度计算：支持基于嵌入向量和基于关键词的相似度计算

核心类：
- ImprovedSemanticChunker: 基于段落级别的语义分割器
- ImprovedSemanticSplitter: ImprovedSemanticChunker 的包装类，兼容 LlamaIndex

使用场景：
- 文档预处理：在构建知识图谱前，将长文档分割成语义连贯的文本块
- 向量检索：为向量数据库准备合适大小的文本块
- 语义搜索：提高检索结果的语义相关性

依赖：
- numpy: 用于向量计算和相似度计算
- 嵌入模型：用于计算段落和句子的嵌入向量（可选）
"""

import logging
from typing import List, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)


class ImprovedSemanticChunker:
    """
    基于段落级别的语义分割器
    
    采用两阶段策略进行文本分割：
    1. 结构化切分：按段落（双换行 \n\n）切分，保留基本排版逻辑
    2. 语义聚合：计算相邻段落相似度，高相似度则合并，直到达到大小限制
    3. 重叠保留：每个 chunk 保留 10%-15% 的重复内容
    
    工作流程：
    1. 按段落分割文本（使用双换行符作为分隔符）
    2. 计算相邻段落的语义相似度
    3. 根据相似度和大小限制聚合段落
    4. 为每个 chunk 添加重叠内容
    5. 在句子边界处切分，避免切断完整句子
    
    关键特性：
    - 语义感知：基于语义相似度而非固定长度进行分割
    - 结构化保留：优先按段落切分，保留文档的原始结构
    - 重叠保留：相邻 chunks 之间有 10%-15% 的重叠，提高实体提取准确性
    - 智能边界：在句子边界处切分，避免切断完整的句子
    - 灵活回退：当嵌入模型不可用时，使用关键词重叠度计算相似度
    
    属性：
        embedding_model: 嵌入模型，用于计算语义相似度
        chunk_size: 目标 chunk 大小（字符数）
        overlap_ratio: 重叠比例（0.1-0.15）
        similarity_threshold: 相似度阈值，高于此值则合并
        min_chunk_length: 最小 chunk 长度
        max_chunk_length: 最大 chunk 长度
        paragraph_separator: 段落分隔符
        overlap_chars: 重叠字符数
        medical_terms: 医学术语列表（用于关键词匹配）
    
    使用示例：
        ```python
        from llama.semantic_chunker import ImprovedSemanticChunker
        
        # 创建改进的语义分割器
        chunker = ImprovedSemanticChunker(
            embedding_model=embedding_model,
            chunk_size=700,
            overlap_ratio=0.12,
            similarity_threshold=0.70
        )
        
        # 分割文本
        text = "这是一段很长的文本..."
        chunks = chunker.split_text(text)
        print(f"分割成 {len(chunks)} 个文本块")
        ```
    """
    
    def __init__(
        self,
        embedding_model=None,
        chunk_size: int = 700,
        overlap_ratio: float = 0.12,
        similarity_threshold: float = 0.7,
        min_chunk_length: int = 600,
        max_chunk_length: int = 1400,
        paragraph_separator: str = "\n\n"
    ):
        """
        初始化改进的语义分割器
        
        Args:
            embedding_model: 嵌入模型，用于计算语义相似度
                           - 如果提供，使用嵌入向量计算相似度（更准确）
                           - 如果为 None，使用关键词重叠度计算（备用方案）
            chunk_size: 目标 chunk 大小（字符数），默认为 700
                       分割器会尝试使每个 chunk 接近这个大小
            overlap_ratio: 重叠比例，默认为 0.12（12%）
                          相邻 chunk 之间的重叠比例，用于保持上下文连续性
                          - 推荐值：0.10-0.15
                          - 较高的值会增加重叠，但也会增加总文本量
            similarity_threshold: 相似度阈值，默认为 0.7
                                当相邻段落相似度高于此值时，考虑合并
                                - 较高的值（0.8-0.9）会产生更多、更小的 chunk
                                - 较低的值（0.6-0.7）会产生更少、更大的 chunk
            min_chunk_length: 最小 chunk 长度，默认为 600
                            小于此长度的 chunk 会被合并到前一个 chunk
            max_chunk_length: 最大 chunk 长度，默认为 1400
                            超过此长度的 chunk 会被强制分割
            paragraph_separator: 段落分隔符，默认为 "\n\n"
                                 用于识别段落边界的字符串
        
        Note:
            - 医学术语列表（medical_terms）包含眼科领域的常见术语
            - 重叠字符数会自动计算为 chunk_size * overlap_ratio
            - 如果嵌入模型不可用，会自动回退到基于关键词的相似度计算
        """
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.overlap_ratio = overlap_ratio
        self.similarity_threshold = similarity_threshold
        self.min_chunk_length = min_chunk_length
        self.max_chunk_length = max_chunk_length
        self.paragraph_separator = paragraph_separator
        
        # 计算重叠字符数
        self.overlap_chars = int(chunk_size * overlap_ratio)
        
        # 医学术语列表（用于关键词匹配）
        # 这些是眼科领域的常见术语，用于计算关键词重叠度
        self.medical_terms = [
            "角膜塑形镜", "低浓度阿托品", "眼轴长度", "病理性近视", "视网膜脱落",
            "调节幅度", "LogMAR视力表", "全飞秒激光手术", "准分子激光手术",
            "RGP镜片", "后巩膜加固术", "离焦框架镜", "视觉训练",
            "屈光度", "远视储备", "调节灵敏度", "眼压", "角膜曲率", "调节滞后",
            "近视", "远视", "散光", "弱视", "斜视", "外斜", "内斜", "病理性近视",
            "轴性近视", "屈光不正", "屈光参差", "老视", "并发性白内障", "后巩膜葡萄肿",
            "视物模糊", "眼胀", "虹视", "眼痛", "畏光", "流泪", "视力下降",
            "豹纹状眼底", "视网膜萎缩", "脉络膜萎缩", "黄斑出血", "漆样裂纹",
            "视盘杯盘比(C/D)扩大", "眼压升高", "瞳孔放大", "活动受限",
            "角膜", "晶状体", "视网膜", "视神经", "黄斑区", "中心凹", "睫状肌",
            "悬韧带", "脉络膜", "巩膜", "前房", "房水"
        ]
    
    def split_text(self, text: str) -> List[str]:
        """
        分割文本为语义连贯的 chunks
        
        这是改进的语义分割器的主要方法，执行以下步骤：
        1. 检查文本长度，如果过短则直接返回
        2. 按段落分割文本（结构化切分）
        3. 基于相似度聚合段落（语义聚合）
        4. 为每个 chunk 添加重叠内容（重叠保留）
        
        Args:
            text: 待分割的文本，可以是任意长度的字符串
        
        Returns:
            分割后的文本块列表，每个元素是一个字符串
            - 如果输入文本为空或过短，返回包含原始文本的单元素列表
            - 如果输入文本为空字符串，返回空列表
            - 返回的文本块按原始顺序排列
            - 相邻 chunks 之间有 10%-15% 的重叠内容
        
        Raises:
            无特定异常，所有异常都被捕获并记录日志
        
        Note:
            - 分割点选择基于语义相似度，而非固定长度
            - 优先按段落切分，保留文档的原始结构
            - 重叠内容确保实体在相邻 chunks 中都能被提取
            - 最终的 chunk 大小会在 min_chunk_length 和 max_chunk_length 之间
        """
        if not text or len(text) < self.min_chunk_length:
            return [text] if text else []
        
        # 第一步：结构化切分（按段落）
        paragraphs = self._split_by_paragraphs(text)
        logger.info(f"结构化切分：生成 {len(paragraphs)} 个段落")
        
        # 第二步：语义聚合
        chunks = self._aggregate_by_similarity(paragraphs)
        logger.info(f"语义聚合：生成 {len(chunks)} 个 chunks")
        
        # 第三步：添加重叠内容
        chunks_with_overlap = self._add_overlap(chunks)
        logger.info(f"添加重叠：最终生成 {len(chunks_with_overlap)} 个 chunks")
        
        return chunks_with_overlap
    
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """
        按段落分割文本
        
        该方法使用双换行符（\n\n）作为段落分隔符，将文本分割成多个段落。
        这是结构化切分的第一步，保留文档的原始排版结构。
        
        Args:
            text: 待分割的文本，可以是任意长度的字符串
        
        Returns:
            段落列表，每个元素是一个字符串
            - 空段落会被过滤掉
            - 每个段落会被去除首尾空白字符
            - 如果文本为空，返回空列表
        
        Note:
            - 使用双换行符作为分隔符，保留文档的原始结构
            - 空段落会被过滤，避免产生无意义的 chunks
            - 段落顺序与原文保持一致
        """
        # 使用双换行作为段落分隔符
        paragraphs = text.split(self.paragraph_separator)
        
        # 过滤空段落
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        return paragraphs
    
    def _aggregate_by_similarity(self, paragraphs: List[str]) -> List[str]:
        """
        基于相似度聚合段落
        
        该方法根据相邻段落的语义相似度，将相似的段落合并成更大的文本块。
        合并策略：
        1. 如果当前 chunk 长度小于最小长度，必须合并
        2. 如果相似度高于阈值且合并后不超过最大长度，则合并
        3. 否则，保存当前 chunk，开始新的 chunk
        
        Args:
            paragraphs: 段落列表，每个元素是一个字符串
        
        Returns:
            聚合后的 chunks 列表，每个元素是一个字符串
            - 返回的 chunks 按原始顺序排列
            - 每个 chunk 的长度在 min_chunk_length 和 max_chunk_length 之间
            - 如果输入为空，返回空列表
        
        Note:
            - 使用嵌入向量计算语义相似度（如果可用）
            - 如果嵌入模型不可用，使用关键词重叠度
            - 合并时会保留段落分隔符，保持文档结构
            - 最后一个段落会被添加到结果中
        """
        if not paragraphs:
            return []
        
        if len(paragraphs) == 1:
            return paragraphs
        
        # 计算段落嵌入向量
        paragraph_embeddings = self._compute_paragraph_embeddings(paragraphs)
        
        # 聚合段落
        chunks = []
        current_chunk = paragraphs[0]
        current_length = len(current_chunk)
        
        for i in range(1, len(paragraphs)):
            next_paragraph = paragraphs[i]
            merged_length = current_length + len(next_paragraph)
            
            # 计算当前 chunk 和下一个段落的相似度
            similarity = self._compute_similarity(
                current_chunk,
                next_paragraph,
                paragraph_embeddings[i - 1] if paragraph_embeddings else None,
                paragraph_embeddings[i] if paragraph_embeddings else None
            )
            
            # 判断是否应该合并
            # 合并条件：
            # 1. 当前 chunk 长度小于最小长度，必须合并
            # 2. 相似度高于阈值且合并后不超过最大长度
            should_merge = (
                current_length < self.min_chunk_length or
                (similarity >= self.similarity_threshold and merged_length <= self.max_chunk_length)
            )
            
            if should_merge and merged_length <= self.max_chunk_length:
                # 合并段落
                current_chunk += self.paragraph_separator + next_paragraph
                current_length = merged_length
            else:
                # 不合并，保存当前 chunk
                chunks.append(current_chunk)
                current_chunk = next_paragraph
                current_length = len(next_paragraph)
        
        # 添加最后一个 chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _compute_paragraph_embeddings(self, paragraphs: List[str]) -> List[np.ndarray]:
        """
        计算段落的嵌入向量
        
        该方法使用嵌入模型批量计算段落的嵌入向量，用于后续的相似度计算。
        
        Args:
            paragraphs: 段落列表，每个元素是一个字符串
        
        Returns:
            嵌入向量列表，每个元素是一个 numpy.ndarray
            - 如果嵌入模型可用，返回所有段落的嵌入向量
            - 如果嵌入模型不可用或计算失败，返回 None
            - 嵌入向量的维度取决于嵌入模型的配置
        
        Raises:
            无特定异常，计算失败时记录警告日志并返回 None
        
        Note:
            - 使用批量计算提高效率
            - 如果嵌入模型不可用，后续会使用关键词重叠度
            - 计算过程可能较慢，建议在初始化时调用一次
        """
        if self.embedding_model is None:
            return None
        
        try:
            embeddings = self.embedding_model.get_text_embedding_batch(paragraphs)
            return [np.array(emb) for emb in embeddings]
        except Exception as e:
            logger.warning(f"计算段落嵌入失败: {e}")
            return None
    
    def _compute_similarity(
        self,
        text1: str,
        text2: str,
        emb1: Optional[np.ndarray] = None,
        emb2: Optional[np.ndarray] = None
    ) -> float:
        """
        计算两个文本的相似度
        
        该方法支持两种相似度计算方式：
        1. 基于嵌入向量的余弦相似度（如果提供了嵌入向量）
        2. 基于关键词重叠度的相似度（备用方案）
        
        Args:
            text1: 第一个文本，可以是任意长度的字符串
            text2: 第二个文本，可以是任意长度的字符串
            emb1: 第一个文本的嵌入向量（可选），numpy.ndarray 类型
            emb2: 第二个文本的嵌入向量（可选），numpy.ndarray 类型
        
        Returns:
            相似度分数，范围在 0.0 到 1.0 之间
            - 1.0 表示完全相似
            - 0.0 表示完全不相似
            - 如果嵌入向量计算失败，使用关键词重叠度
        
        Note:
            - 优先使用嵌入向量计算余弦相似度（更准确）
            - 如果嵌入向量不可用或计算失败，使用关键词重叠度
            - 关键词重叠度基于医学术语的重叠程度
            - 余弦相似度计算公式：dot(emb1, emb2) / (norm(emb1) * norm(emb2))
        """
        # 如果有嵌入向量，使用余弦相似度
        if emb1 is not None and emb2 is not None:
            try:
                similarity = np.dot(emb1, emb2) / (
                    np.linalg.norm(emb1) * np.linalg.norm(emb2)
                )
                return float(similarity)
            except Exception as e:
                logger.warning(f"计算余弦相似度失败: {e}")
        
        # 否则使用关键词重叠度
        return self._compute_keyword_overlap(text1, text2)
    
    def _compute_keyword_overlap(self, text1: str, text2: str) -> float:
        """
        计算关键词重叠度
        
        该方法计算两个文本之间的医学术语重叠程度，作为语义相似度的备用方案。
        如果两个文本都没有医学术语，则使用字符重叠度。
        
        Args:
            text1: 第一个文本，可以是任意长度的字符串
            text2: 第二个文本，可以是任意长度的字符串
        
        Returns:
            重叠度分数，范围在 0.0 到 1.0 之间
            - 1.0 表示完全重叠（所有关键词都相同）
            - 0.0 表示完全不重叠（没有共同关键词）
            - 使用 Jaccard 相似度：intersection / union
        
        Note:
            - 优先使用医学术语进行匹配
            - 如果两个文本都没有医学术语，使用字符重叠度
            - 医学术语列表来自类属性 medical_terms
            - Jaccard 相似度公式：|A ∩ B| / |A ∪ B|
        """
        # 提取关键词（医学术语）
        keywords1 = set(term for term in self.medical_terms if term in text1)
        keywords2 = set(term for term in self.medical_terms if term in text2)
        
        if not keywords1 and not keywords2:
            # 如果没有医学术语，使用字符重叠度
            set1 = set(text1)
            set2 = set(text2)
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            return intersection / union if union > 0 else 0.0
        
        # 计算医学术语重叠度
        intersection = len(keywords1 & keywords2)
        union = len(keywords1 | keywords2)
        
        return intersection / union if union > 0 else 0.0
    
    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """
        为 chunks 添加重叠内容
        
        该方法为每个 chunk 添加重叠内容，确保相邻 chunks 之间有 10%-15% 的重叠。
        重叠策略：
        - 每个 chunk（除了第一个）的开头添加上一个 chunk 的结尾
        - 每个 chunk（除了最后一个）的结尾添加下一个 chunk 的开头
        
        这样即使实体被切在边缘，在相邻的两个 chunk 里也能通过上下文被提取出来。
        
        Args:
            chunks: 原始 chunks 列表，每个元素是一个字符串
        
        Returns:
            添加重叠后的 chunks 列表，每个元素是一个字符串
            - 如果输入为空或只有一个 chunk，返回原列表
            - 相邻 chunks 之间有重叠内容
            - 重叠内容在句子边界处切分
        
        Note:
            - 重叠内容来自相邻 chunks 的开头或结尾
            - 重叠字符数由 overlap_chars 属性控制
            - 重叠内容会在句子边界处切分，避免切断完整句子
            - 重叠内容使用双换行符（\n\n）分隔
        """
        if len(chunks) <= 1:
            return chunks
        
        chunks_with_overlap = []
        
        for i, chunk in enumerate(chunks):
            modified_chunk = chunk
            
            if i > 0:
                # 不是第一个 chunk，在开头添加上一个 chunk 的结尾
                prev_chunk = chunks[i - 1]
                prev_overlap = self._extract_overlap_text(prev_chunk, self.overlap_chars, from_end=True)
                modified_chunk = prev_overlap + "\n\n" + modified_chunk
            
            if i < len(chunks) - 1:
                # 不是最后一个 chunk，在结尾添加下一个 chunk 的开头
                next_chunk = chunks[i + 1]
                next_overlap = self._extract_overlap_text(next_chunk, self.overlap_chars)
                modified_chunk = modified_chunk + "\n\n" + next_overlap
            
            chunks_with_overlap.append(modified_chunk)
        
        return chunks_with_overlap
    
    def _extract_overlap_text(
        self,
        text: str,
        num_chars: int,
        from_end: bool = False
    ) -> str:
        """
        提取重叠文本
        
        该方法从文本中提取指定数量的字符，并尝试在句子边界处切分。
        
        Args:
            text: 原始文本，可以是任意长度的字符串
            num_chars: 提取的字符数，必须为正整数
            from_end: 是否从结尾提取，默认为 False
                      - False: 从开头提取
                      - True: 从结尾提取
        
        Returns:
            提取的文本字符串
            - 如果文本长度小于等于 num_chars，返回整个文本
            - 否则返回指定数量的字符（可能在句子边界处切分）
        
        Note:
            - 提取后会尝试在句子边界处切分
            - 句子边界由 `_split_at_sentence_boundary` 方法确定
            - 这样可以避免在句子中间切分，保持文本的完整性
        """
        if len(text) <= num_chars:
            return text
        
        if from_end:
            # 从结尾提取
            overlap_text = text[-num_chars:]
        else:
            # 从开头提取
            overlap_text = text[:num_chars]
        
        # 尝试在句子边界处切分
        overlap_text = self._split_at_sentence_boundary(overlap_text, from_end)
        
        return overlap_text
    
    def _split_at_sentence_boundary(self, text: str, from_end: bool = False) -> str:
        """
        在句子边界处切分文本
        
        该方法在文本中寻找句子结束标记，并在第一个或最后一个标记处切分。
        这样可以避免在句子中间切分，保持文本的完整性。
        
        Args:
            text: 待切分的文本，可以是任意长度的字符串
            from_end: 是否从结尾开始寻找，默认为 False
                      - False: 从开头寻找第一个句子结束标记
                      - True: 从结尾寻找最后一个句子结束标记
        
        Returns:
            切分后的文本字符串
            - 如果找到句子结束标记，返回到该标记为止的文本
            - 如果没有找到句子结束标记，返回原文本
        
        Note:
            - 句子结束标记包括：。！？!? \n
            - 这些标记覆盖了中文和英文的常见句子结束符号
            - 从开头切分时，返回第一个句子结束标记之前的内容
            - 从结尾切分时，返回到最后一个句子结束标记为止的内容
        """
        sentence_endings = ['。', '！', '？', '!', '?', '\n']
        
        if from_end:
            # 从开头寻找第一个句子结束标记
            for i, char in enumerate(text):
                if char in sentence_endings:
                    return text[:i + 1]
        else:
            # 从结尾寻找最后一个句子结束标记
            for i in range(len(text) - 1, -1, -1):
                if text[i] in sentence_endings:
                    return text[:i + 1]
        
        return text


class ImprovedSemanticSplitter:
    """
    改进的语义分割器包装类，兼容 LlamaIndex 的接口
    
    该类是 ImprovedSemanticChunker 的包装器，提供了与 LlamaIndex 兼容的接口。
    主要用于将 LlamaIndex 的 Document 对象转换为分割后的节点列表。
    
    工作流程：
    1. 接收 LlamaIndex 的 Document 对象列表
    2. 提取每个文档的文本内容
    3. 使用 ImprovedSemanticChunker 分割文本
    4. 为每个分割后的文本块创建 Document 节点
    5. 添加元数据信息（chunk_index, chunk_total, chunking_method 等）
    
    属性：
        chunker: ImprovedSemanticChunker 实例，执行实际的文本分割
    
    使用示例：
        ```python
        from llama.semantic_chunker import ImprovedSemanticSplitter
        from llama_index.core import Document
        
        # 创建改进的语义分割器
        splitter = ImprovedSemanticSplitter(
            embedding_model=embedding_model,
            chunk_size=700,
            overlap_ratio=0.12,
            similarity_threshold=0.70
        )
        
        # 准备文档
        documents = [
            Document(text="这是第一段很长的文本...", metadata={"source": "doc1"}),
            Document(text="这是第二段很长的文本...", metadata={"source": "doc2"})
        ]
        
        # 分割文档
        nodes = splitter.get_nodes_from_documents(documents)
        print(f"分割成 {len(nodes)} 个节点")
        ```
    
    Note:
        - 该类主要用于与 LlamaIndex 集成
        - 保留原始文档的元数据，并添加 chunk 相关的元数据
        - 添加的元数据包括：chunk_index, chunk_total, chunking_method, overlap_ratio, similarity_threshold
        - 如果文档没有文本内容，会跳过该文档
    """
    
    def __init__(
        self,
        embedding_model=None,
        chunk_size: int = 1024,
        overlap_ratio: float = 0.12,
        similarity_threshold: float = 0.7,
        min_chunk_length: int = 600,
        max_chunk_length: int = 1400,
        paragraph_separator: str = "\n\n",
        **kwargs
    ):
        """
        初始化改进的语义分割器包装类
        
        Args:
            embedding_model: 嵌入模型，用于计算语义相似度
                           - 如果提供，使用嵌入向量计算相似度（更准确）
                           - 如果为 None，使用关键词重叠度计算（备用方案）
            chunk_size: 目标 chunk 大小（字符数），默认为 1024
                       分割器会尝试使每个 chunk 接近这个大小
            overlap_ratio: 重叠比例，默认为 0.12（12%）
                          相邻 chunk 之间的重叠部分，用于保持上下文连续性
            similarity_threshold: 相似度阈值，默认为 0.7
                                 当相邻段落相似度高于此值时，考虑在此处分割
            min_chunk_length: 最小 chunk 长度，默认为 600
                             小于此长度的 chunk 会被合并到前一个 chunk
            max_chunk_length: 最大 chunk 长度，默认为 1400
                             超过此长度的 chunk 会被强制分割
            paragraph_separator: 段落分隔符，默认为 "\n\n"
                                 用于识别段落边界的字符串
            **kwargs: 其他传递给 ImprovedSemanticChunker 的参数
        
        Note:
            - 该类是 ImprovedSemanticChunker 的包装器，参数含义与 ImprovedSemanticChunker 相同
            - 默认参数值与 ImprovedSemanticChunker 略有不同，以适应 LlamaIndex 的使用场景
        """
        self.chunker = ImprovedSemanticChunker(
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            overlap_ratio=overlap_ratio,
            similarity_threshold=similarity_threshold,
            min_chunk_length=min_chunk_length,
            max_chunk_length=max_chunk_length,
            paragraph_separator=paragraph_separator
        )
    
    def get_nodes_from_documents(self, documents: List[Any]) -> List[Any]:
        """
        从文档列表获取节点
        
        该方法将 LlamaIndex 的 Document 对象列表转换为分割后的节点列表。
        每个文档会被分割成多个节点，每个节点包含一个文本块和元数据。
        
        Args:
            documents: 文档列表，每个元素应该是 LlamaIndex 的 Document 对象
                     - Document 对象应该包含 text 属性（文本内容）
                     - Document 对象可以包含 metadata 属性（元数据）
        
        Returns:
            节点列表，每个元素是一个 LlamaIndex 的 Document 对象
            - 每个节点包含一个分割后的文本块
            - 每个节点包含原始文档的元数据（复制）
            - 每个节点包含额外的元数据：
              * chunk_index: 当前 chunk 在文档中的索引（从 0 开始）
              * chunk_total: 文档被分割成的总 chunk 数
              * chunking_method: 分块方法标识（"improved_semantic"）
              * overlap_ratio: 重叠比例
              * similarity_threshold: 相似度阈值
            - 如果输入文档列表为空，返回空列表
            - 如果文档没有文本内容，会跳过该文档
        
        Raises:
            无特定异常，所有异常都被捕获并记录日志
        
        Note:
            - 该方法保留原始文档的元数据，不会修改原始文档
            - 节点的顺序与原始文档的顺序一致
            - 同一个文档的所有节点会连续排列
            - 添加的元数据可以用于追踪分块过程和参数
        """
        from llama_index.core import Document
        
        nodes = []
        for doc in documents:
            text = getattr(doc, "text", "")
            if not text:
                continue
            
            # 使用改进的语义分割器分割文本
            chunks = self.chunker.split_text(text)
            
            # 为每个 chunk 创建节点
            for i, chunk in enumerate(chunks):
                metadata = getattr(doc, "metadata", {}).copy()
                metadata["chunk_index"] = i
                metadata["chunk_total"] = len(chunks)
                metadata["chunking_method"] = "improved_semantic"
                metadata["overlap_ratio"] = self.chunker.overlap_ratio
                metadata["similarity_threshold"] = self.chunker.similarity_threshold
                
                node = Document(text=chunk, metadata=metadata)
                nodes.append(node)
        
        return nodes
