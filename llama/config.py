"""
知识图谱构建配置模块

该模块提供了知识图谱构建所需的所有配置，包括：
- API 配置（SiliconFlow）
- Neo4j 数据库配置
- 文档处理配置
- 实体提取器配置
- 嵌入模型配置
- OSS 存储配置（腾讯云 COS）
- 重排序配置（Rerank）
- 实体描述生成配置
- 请求限流配置

所有配置都支持通过环境变量进行覆盖，便于在不同环境中部署。

使用示例：
    ```python
    from llama.config import API_CONFIG, DOCUMENT_CONFIG, NEO4J_CONFIG
    
    # 获取 API 配置
    api_key = API_CONFIG["siliconflow"]["api_key"]
    
    # 获取文档配置
    chunk_size = DOCUMENT_CONFIG["chunk_size"]
    ```
"""

import os
import logging
import ssl

# 全局禁用 SSL 验证
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# 禁用 urllib3 的警告
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from logging.handlers import TimedRotatingFileHandler
import tempfile
from datetime import datetime
from typing import Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# 局部禁用代理，确保 localhost 和 127.0.0.1 不走代理，同时避免过度禁用导致 DNS 解析失败
os.environ['no_proxy'] = 'localhost,127.0.0.1'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'

# 加载 .env 文件
load_dotenv()

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 日志配置
_logging_initialized = False


def setup_logging(log_dir: str = None) -> logging.Logger:
    """
    设置日志配置，按日期生成日志文件
    
    该函数配置全局日志系统，支持自定义日志目录。
    如果指定的日志目录不可用，会自动回退到默认目录或系统临时目录。
    
    Args:
        log_dir: 日志目录路径，默认为项目根目录下的 logs 文件夹
                 可以通过环境变量 LOG_DIR 进行配置
                 - 如果是绝对路径，直接使用
                 - 如果是相对路径或 None，使用当前工作目录下的 logs 文件夹
    
    Returns:
        logging.Logger: 配置好的日志记录器实例
        
    Note:
        - 使用单例模式，同一进程内只初始化一次
        - 日志文件按日期命名，格式为：llama_index_YYYYMMDD.log
        - 同时输出到文件和控制台
        - 如果指定目录不可写，会尝试回退到默认目录
        - 如果默认目录也不可写，会回退到系统临时目录
        
    Raises:
        无特定异常，所有异常都被捕获并记录
    """
    global _logging_initialized
    
    # 防止同一进程内重复初始化
    if _logging_initialized:
        return logging.getLogger(__name__)
    
    _logging_initialized = True
    
    # 确定日志目录
    if log_dir is None:
        env_log_dir = os.getenv("LOG_DIR")
        if env_log_dir and os.path.isabs(env_log_dir):
            log_dir = env_log_dir
        else:
            log_dir = str(Path(os.getcwd()) / "logs")
    
    # 路径验证与回退机制
    try:
        # 1. 尝试创建目录
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        # 2. 验证目录是否可写
        test_file = os.path.join(log_dir, f".write_test_{os.getpid()}")
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        
    except Exception as e:
        # 如果指定路径不可用，尝试回退到默认路径（当前工作目录）
        fallback_dir = str(Path(os.getcwd()) / "logs")
        print(f"Warning: Log directory '{log_dir}' is not accessible: {e}")
        print(f"Attempting fallback to default: {fallback_dir}")
        
        try:
            log_dir = fallback_dir
            if not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            # 再次验证
            test_file = os.path.join(log_dir, f".write_test_{os.getpid()}")
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except Exception as e2:
            # 如果默认路径也不可用，回退到系统临时目录
            log_dir = tempfile.gettempdir()
            print(f"Warning: Default log directory also failed: {e2}")
            print(f"Using system temp directory: {log_dir}")

    # 自定义按天滚动的 Handler，保证文件名始终包含当前日期
    class DailyDateFilenameHandler(logging.FileHandler):
        def __init__(self, log_dir, prefix="llama_index"):
            self.log_dir = log_dir
            self.prefix = prefix
            self.current_date = datetime.now().strftime("%Y-%m-%d")
            filename = os.path.join(log_dir, f"{prefix}_{self.current_date}.log")
            super().__init__(filename, encoding='utf-8')
            
        def emit(self, record):
            # 检查日期是否变更
            new_date = datetime.now().strftime("%Y-%m-%d")
            if new_date != self.current_date:
                self.current_date = new_date
                # 关闭当前流
                self.acquire()
                try:
                    self.stream.close()
                    # 更新文件名并重新打开
                    self.baseFilename = os.path.join(self.log_dir, f"{self.prefix}_{self.current_date}.log")
                    self.stream = self._open()
                finally:
                    self.release()
            super().emit(record)

    # 使用自定义 Handler
    # 清除现有的处理器（避免重复日志）
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    file_handler = DailyDateFilenameHandler(log_dir)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[file_handler, logging.StreamHandler()]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Log directory initialized at: {log_dir}")
    
    return logger


# API 配置 - 支持通过环境变量配置
API_CONFIG = {
    "siliconflow": {
        "api_key": os.getenv("SILICONFLOW_API_KEY"),
        "llm_model": os.getenv("SILICONFLOW_LLM_MODEL"),
        "lightweight_model": os.getenv("SILICONFLOW_LIGHTWEIGHT_MODEL"),
        "embedding_model": os.getenv("SILICONFLOW_EMBEDDING_MODEL"),
        "timeout": int(os.getenv("SILICONFLOW_TIMEOUT", "120")),
        "max_tokens": int(os.getenv("SILICONFLOW_MAX_TOKENS", "8192")),
        "max_retries": int(os.getenv("SILICONFLOW_MAX_RETRIES", "3")),
        "temperature": float(os.getenv("SILICONFLOW_TEMPERATURE", "0.0")),
        "ocr_model": os.getenv("SILICONFLOW_OCR_MODEL")
    }
}


# Neo4j 配置 - 支持通过环境变量配置
NEO4J_CONFIG = {
    "username": os.getenv("NEO4J_USERNAME"),
    "password": os.getenv("NEO4J_PASSWORD"),
    "url": os.getenv("NEO4J_URL"),
    "database": os.getenv("NEO4J_DATABASE")
}


# 文档处理配置
DOCUMENT_CONFIG = {
    "path": os.getenv("DOCUMENT_PATH", str(PROJECT_ROOT / "data")),
    "supported_extensions": [".txt", ".docx", ".pdf", ".md"],
    "max_paths_per_chunk": int(os.getenv("MAX_PATHS_PER_CHUNK", "2")),
    "num_workers": int(os.getenv("DOCUMENT_NUM_WORKERS", "4")),
    "chunk_size": int(os.getenv("DOC_CHUNK_SIZE", "1000")),
    "CHUNK_OVERLAP": int(os.getenv("DOC_CHUNK_OVERLAP", "200")),
    "max_chunk_length": int(os.getenv("DOC_MAX_CHUNK_LENGTH", "1400")),
    "min_chunk_length": int(os.getenv("DOC_MIN_CHUNK_LENGTH", "500")),
    "dynamic_chunking": os.getenv("DOC_DYNAMIC_CHUNKING", "true").lower() == "true",
    "dynamic_target_chars_per_chunk": int(os.getenv("DOC_TARGET_CHARS_PER_CHUNK", "1200")),
    "benchmark_chunking": os.getenv("DOC_BENCHMARK_CHUNKING", "false").lower() == "true",
    "log_chunk_metrics": os.getenv("DOC_LOG_CHUNK_METRICS", "true").lower() == "true",
    "sentence_splitter": os.getenv("DOC_SENTENCE_SPLITTER", "。！？!?"),
    "semantic_separator": os.getenv("DOC_SEMANTIC_SEPARATOR", "\n\n"),
    "use_semantic_chunking": os.getenv("USE_SEMANTIC_CHUNKING", "true").lower() == "true",
    "similarity_threshold": float(os.getenv("SEMANTIC_SIMILARITY_THRESHOLD", "0.75")),
    "incremental_processing": os.getenv("INCREMENTAL_PROCESSING", "true").lower() == "true",
    "batch_size": int(os.getenv("DOC_BATCH_SIZE", "10")),
    "progress_update_every_batches": int(os.getenv("PROGRESS_UPDATE_EVERY_BATCHES", "3")),
    "use_multithreading_chunking": os.getenv("USE_MULTITHREADING_CHUNKING", "true").lower() == "true",
    "max_chunking_workers": int(os.getenv("MAX_CHUNKING_WORKERS", "4")),
}


# 实体提取器配置
EXTRACTOR_CONFIG = {
    "max_triplets_per_chunk": int(os.getenv("MAX_TRIPLETS_PER_CHUNK", "10")),
    "num_workers": int(os.getenv("EXTRACTOR_NUM_WORKERS", "10")),
    "min_entities_per_chunk": int(os.getenv("MIN_ENTITIES_PER_CHUNK", "0")),
    "entity_length_limit": int(os.getenv("ENTITY_LENGTH_LIMIT", "25")),
    "allow_non_standard_entities": os.getenv("ALLOW_NON_STANDARD_ENTITIES", "true").lower() == "true",
    "entity_confidence_threshold": float(os.getenv("ENTITY_CONFIDENCE_THRESHOLD", "0.7")),
    "extract_prompt": """# Role: 眼科视光知识图谱构建专家

## 核心任务  
从眼科视光医学文本中提取**具有高度临床价值**的实体和关系，生成结构化的"实体-关系-实体"三元组，用于Neo4j知识图谱构建。

### 1. 分层抽取策略
- **第一层**：实体识别（7大类别）
- **第二层**：关系分类（10种关系类型）
- **第三层**：质量验证（自检机制）

### 2. 实体类别定义（增强版）
1. **疾病与异常** (Disease): 白内障、沙眼、青光眼、低视力、盲、病理性近视、老年性黄斑变性
2. **部位与生理** (Anatomy): 晶状体、视网膜、房水、眼压、视神经、黄斑区、角膜
3. **诊疗与干预** (Treatment): 白内障囊外摘除术、抗生素、视力筛查、验光、OK镜、人工晶体植入
4. **症状与体征** (Symptom): 视力下降、视野缺损、眼红、畏光、复视
5. **流行病学与统计** (Epidemiology):
   - 研究方法：横断面研究、队列研究、病例对照研究、随机对照试验(RCT)、双盲法
   - 统计指标：发病率、患病率、相对危险度(RR)、比值比(OR)、灵敏度、特异度、Kappa值
6. **卫生管理与经济** (Healthcare):
   - 评价指标：成本-效果分析(CEA)、质量调整生命年(QALY)、伤残调整生命年(DALY)
   - 策略与行动：一级预防、初级眼保健、视觉2020行动、防盲治盲
7. **风险因素** (RiskFactor): 年龄、紫外线辐射、吸烟、遗传因素、糖尿病、高血压

### 3. 关系类型定义（精确映射）
关系类型	方向	说明
包含	A → B	A包含B（如：流行病学研究 → 观察性研究）
采用	A → B	A采用B作为方法（如：防盲工作 → 快速评估法）
评估	A → B	A用于评估B（如：Kappa值 → 一致性）
导致	A → B	A导致B（如：紫外线 → 白内障）
治疗	A → B	A治疗B（如：白内障手术 → 白内障）
属于	A → B	A属于B分类（如：随机对照试验 → 实验性研究）
指标为	A → B	A的指标是B（如：卫生经济学评价 → 成本-效果比）
关联	A ↔ B	医学术语与疾病关联（如：屈光度 ↔ 近视）
影响	A → B	疾病影响解剖结构（如：青光眼 → 视神经）
风险因素为	A → B	A是B的风险因素（如：糖尿病 → 白内障）
预防策略	A → B	对A采用B策略（如：沙眼 → SAFE战略）
重点防治	A → B	A项目重点防治B（如：视觉2020 → 白内障）

### 4. 质量约束与去噪
**必须剔除的实体**：
- ❌ 行政信息：人名（易虹主任）、日期（2018年）、地点（重庆）、机构名
- ❌ 代词与泛指：我们、患者、家长、学生、眼睛（太泛）、问题、方法
- ❌ 纯修饰词：严重的、早期的（需与名词结合，如"严重病理性近视"）

**必须保留的修饰**：
- ✅ 程度修饰：严重病理性近视、早期白内障
- ✅ 具体数值：-6.00D（作为尾实体或属性）

### 5. 自验证机制（Critical Addition）
在提取完成后，模型必须进行自检：
自检清单：
所有实体是否都属于7大类别？
所有关系是否都在10种关系类型中？
是否剔除了行政/代词/泛指实体？
JSON格式是否正确（双引号、无多余逗号）？

## 💡 Few-Shot 示例（覆盖复杂场景）

```json
[
  {
    "head": "队列研究",
    "head_type": "流行病学方法",
    "relation": "属于",
    "tail": "观察性研究",
    "tail_type": "研究类别"
  },
  {
    "head": "紫外线辐射",
    "head_type": "风险因素",
    "relation": "导致",
    "tail": "白内障",
    "tail_type": "疾病"
  },
  {
    "head": "灵敏度",
    "head_type": "统计指标",
    "relation": "评估",
    "tail": "筛查试验",
    "tail_type": "评价对象"
  },
  {
    "head": "成本-效用分析",
    "head_type": "卫生经济学",
    "relation": "指标为",
    "tail": "质量调整生命年",
    "tail_type": "指标"
  },
  {
    "head": "沙眼",
    "head_type": "疾病",
    "relation": "预防策略",
    "tail": "SAFE战略",
    "tail_type": "公共卫生策略"
  },
  {
    "head": "视觉2020",
    "head_type": "行动计划",
    "relation": "重点防治",
    "tail": "白内障",
    "tail_type": "疾病"
  },
  {
    "head": "青光眼",
    "head_type": "疾病",
    "relation": "影响",
    "tail": "视神经",
    "tail_type": "解剖结构"
  },
  {
    "head": "眼压",
    "head_type": "医学术语",
    "relation": "关联",
    "tail": "青光眼",
    "tail_type": "疾病"
  },
  {
    "head": "严重病理性近视",
    "head_type": "疾病",
    "relation": "导致",
    "tail": "视网膜脱落",
    "tail_type": "并发症"
  },
  {
    "head": "OK镜",
    "head_type": "治疗手段",
    "relation": "属于",
    "tail": "硬性角膜接触镜",
    "tail_type": "器具"
  }
]

## 📝 📝 执行指令
1.数量控制：针对当前文本块，提取 3-8个 高质量三元组（最多10个）
2.优先级排序：
    优先：疾病-治疗、疾病-症状、风险因素-疾病
    其次：流行病学方法、统计指标、公共卫生策略
    最后：卫生管理策略
3.格式严格：
    必须使用双引号
    必须包含完整的5个字段（head, head_type, relation, tail, tail_type）
    严禁输出空值或无效三元组

## ⚠️ 最终输出格式

1. **使用双引号**：所有字符串必须使用双引号（"），**严禁使用单引号**
2. **标准JSON结构**：必须符合标准JSON格式，确保所有括号、逗号、冒号正确
3. **完整的三元组**：每个三元组必须包含完整的 head, head_type, relation, tail, tail_type 五个字段
4. **不要输出空值**：不要输出空字符串或null值，如果某个字段确实为空，可以省略该三元组
5. **不要输出无效数据**：不要输出如 {"head": "", "", "", ""} 这样的无效三元组

**正确的JSON格式示例：**
```json
[
  {
    "head": "实体1名称",
    "head_type": "实体1类型",
    "relation": "关系类型",
    "tail": "实体2名称",
    "tail_type": "实体2类型"
  }
]
```

IMPORTANT: Do not output thinking process. Start with [. Expected output: JSON only.
Text: {text}"""
}


# 请求限流配置 (默认值)
RATE_LIMIT_CONFIG = {
    "request_delay": 0.5,           # 请求间隔 (秒)
    "max_concurrent_requests": 5,   # 最大并发
    "retry_delay": 5.0,            # 重试延迟 (秒)
    "rpm_limit": 120,               # 每分钟请求数
    "tpm_limit": 100000,            # 每分钟 Token 数
    "max_tokens_per_request": 2048, # 单次请求最大 Token
    "max_retries": 3                # 最大重试次数
}

# 模型特定频控限制 (基于用户提供数据)
MODEL_RATE_LIMITS = {
    "BAAI/bge-m3": {
        "rpm": 2000,
        "tpm": 500000
    },
    "BAAI/bge-reranker-v2-m3": {
        "rpm": 2000,
        "tpm": 500000
    },
    "Qwen/Qwen2.5-Coder-7B-Instruct": {
        "rpm": 1000,
        "tpm": 50000
    },
    "deepseek-ai/DeepSeek-OCR": {
        "rpm": 1000,
        "tpm": 80000
    }
}

def get_rate_limit(model_name: str) -> dict:
    """
    根据模型名称获取频控参数
    返回字典包含: rpm, tpm, request_delay, retry_delay, max_retries
    """
    limits = MODEL_RATE_LIMITS.get(model_name, {
        "rpm": RATE_LIMIT_CONFIG["rpm_limit"],
        "tpm": RATE_LIMIT_CONFIG["tpm_limit"]
    })
    
    # 动态计算延迟，增加 10% 安全边际
    # delay = 60 / rpm * 1.1
    rpm = limits.get("rpm", RATE_LIMIT_CONFIG["rpm_limit"])
    request_delay = max(0.01, (60.0 / rpm) * 1.1)
    
    return {
        "rpm": rpm,
        "tpm": limits.get("tpm", RATE_LIMIT_CONFIG["tpm_limit"]),
        "request_delay": request_delay,
        "retry_delay": RATE_LIMIT_CONFIG["retry_delay"],
        "max_retries": RATE_LIMIT_CONFIG["max_retries"]
    }


# 嵌入模型配置
EMBEDDING_CONFIG = {
    "use_hybrid_embedding": False,
    "siliconflow_model": "BAAI/bge-m3",
    "local_fallback_model": "BAAI/bge-m3",
    "local_device": "cpu",
    "enable_local_fallback": True,
    "max_consecutive_failures": 1,
    "fallback_timeout": 10,
}


# OSS 配置（腾讯云 COS）
OSS_CONFIG = {
    "drive": "cos",
    "cos_secret_id": os.getenv("COS_SECRET_ID"),
    "cos_secret_key": os.getenv("COS_SECRET_KEY"),
    "cos_bucket": os.getenv("COS_BUCKET"),
    "cos_region": os.getenv("COS_REGION"),
    "cos_path": os.getenv("COS_PATH"),
}


# 重排序配置（Rerank）
RERANK_CONFIG = {
    "enable": os.getenv("RERANK_ENABLE", "true").lower() == "true",
    "provider": os.getenv("RERANK_PROVIDER", "siliconflow"),
    "api_key": os.getenv("RERANK_API_KEY", API_CONFIG["siliconflow"]["api_key"]),
    "model": os.getenv("RERANK_MODEL"),
    "top_n": int(os.getenv("RERANK_TOP_N", "3")),
    "initial_top_k": int(os.getenv("RERANK_INITIAL_TOP_K", "10")),
}

# 混合检索配置（Hybrid Search with RRF using QueryFusionRetriever）
HYBRID_SEARCH_CONFIG = {
    "enable": os.getenv("HYBRID_SEARCH_ENABLE", "true").lower() == "true",
    "initial_top_k": int(os.getenv("HYBRID_SEARCH_INITIAL_TOP_K", "50")),  # 初始检索 Top K（BM25 和 Vector 各自返回的数量）
    "fusion_mode": os.getenv("HYBRID_SEARCH_FUSION_MODE", "reciprocal_rerank"),  # 融合模式：reciprocal_rerank 或 simple
    "num_queries": int(os.getenv("HYBRID_SEARCH_NUM_QUERIES", "1")),  # 查询重写数量（1=不重写，>1=生成多个查询）
    "use_async": os.getenv("HYBRID_SEARCH_USE_ASYNC", "false").lower() == "true",  # 是否异步检索
}


# 实体描述生成配置
ENTITY_DESCRIPTION_CONFIG = {
    "enable": os.getenv("ENTITY_DESCRIPTION_ENABLE", "true").lower() == "true",
    "num_workers": int(os.getenv("ENTITY_DESCRIPTION_NUM_WORKERS", "2")),
    "request_delay": float(os.getenv("ENTITY_DESCRIPTION_REQUEST_DELAY", "0.3")),
    "max_retries": int(os.getenv("ENTITY_DESCRIPTION_MAX_RETRIES", "3")),
    "retry_delay": float(os.getenv("ENTITY_DESCRIPTION_RETRY_DELAY", "5.0")),
    "description_prompt": """你是一名专业的眼科医学专家。请为以下医学实体生成一个简短的医学定义（20-50字）。

实体名称：{entity_name}
实体类型：{entity_type}

要求：
1. 定义要简洁、准确、专业
2. 使用医学专业术语
3. 控制在20-50字之间
4. 只输出定义内容，不要包含任何其他说明或格式

实体定义："""
}


# 任务结果存储（内存中）
task_results: Dict[str, Dict[str, Any]] = {}


# COS 上传器实例（全局）
cos_uploader = None


def initialize_components():
    """
    初始化全局组件
    
    该函数在模块导入时自动调用，初始化 COS 上传器。
    如果初始化失败，会记录错误但不会中断程序执行。
    
    Note:
        - 该函数在模块导入时自动执行
        - 只初始化 COS 上传器，不初始化知识图谱构建器
        - 知识图谱构建器的初始化已移至 main.py
    """
    global cos_uploader
    
    try:
        from llama.oss_uploader import COSUploader, OSSConfig
        oss_config = OSSConfig(OSS_CONFIG)
        cos_uploader = COSUploader(oss_config)
        logger.info("COS上传器初始化成功")
    except Exception as e:
        logger.error(f"COS上传器初始化失败: {e}")
        cos_uploader = None


# 获取 logger 实例
logger = setup_logging()

# 在模块导入时初始化组件
initialize_components()
