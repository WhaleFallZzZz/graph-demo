"""
知识图谱构建配置模块
支持通过环境变量覆盖默认配置
"""

import os
import logging
from typing import Dict, Any
from pathlib import Path

from llama.common import DateTimeUtils

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 日志配置
_logging_initialized = False

def setup_logging(log_dir: str = None) -> logging.Logger:
    """设置日志配置，按日期生成日志文件
    
    Args:
        log_dir: 日志目录路径，默认为项目根目录下的 logs 文件夹
                 可以通过环境变量 LOG_DIR 进行配置
    """
    global _logging_initialized
    
    # 防止同一进程内重复初始化
    if _logging_initialized:
        return logging.getLogger(__name__)
    
    _logging_initialized = True
    if log_dir is None:
        # 优先从环境变量获取；若为绝对路径则使用，否则统一为当前工作目录 ./logs
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
            import tempfile
            log_dir = tempfile.gettempdir()
            print(f"Warning: Default log directory also failed: {e2}")
            print(f"Using system temp directory: {log_dir}")

    # 生成带日期的日志文件名
    current_date = DateTimeUtils.today_str()
    log_file = os.path.join(log_dir, f"llama_index_{current_date}.log")
    
    # 清除现有的处理器（避免重复日志）
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Log directory initialized at: {log_dir}")
    
    return logger

# API配置 - 支持通过环境变量配置
API_CONFIG = {
    "siliconflow": {
        "api_key": os.getenv("SILICONFLOW_API_KEY"),
        "llm_model": os.getenv("SILICONFLOW_LLM_MODEL"),
        "lightweight_model": os.getenv("SILICONFLOW_LIGHTWEIGHT_MODEL"),
        "embedding_model": os.getenv("SILICONFLOW_EMBEDDING_MODEL"),
        "timeout": int(os.getenv("SILICONFLOW_TIMEOUT", "120")),
        "max_tokens": int(os.getenv("SILICONFLOW_MAX_TOKENS", "4096")),
        "max_retries": int(os.getenv("SILICONFLOW_MAX_RETRIES", "3")),
        "temperature": float(os.getenv("SILICONFLOW_TEMPERATURE", "0.0")),
        "ocr_model": os.getenv("SILICONFLOW_OCR_MODEL")
    }
}

# Neo4j配置 - 支持通过环境变量配置
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
    "chunk_size": int(os.getenv("DOC_CHUNK_SIZE", "1024")),
    "CHUNK_OVERLAP": int(os.getenv("DOC_CHUNK_OVERLAP", "120")),
    "max_chunk_length": int(os.getenv("DOC_MAX_CHUNK_LENGTH", "1400")),
    "min_chunk_length": int(os.getenv("DOC_MIN_CHUNK_LENGTH", "600")),
    "dynamic_chunking": os.getenv("DOC_DYNAMIC_CHUNKING", "true").lower() == "true",
    "dynamic_target_chars_per_chunk": int(os.getenv("DOC_TARGET_CHARS_PER_CHUNK", "1200")),
    "benchmark_chunking": os.getenv("DOC_BENCHMARK_CHUNKING", "false").lower() == "true",
    "log_chunk_metrics": os.getenv("DOC_LOG_CHUNK_METRICS", "true").lower() == "true",
    "sentence_splitter": os.getenv("DOC_SENTENCE_SPLITTER", "。！？!?"),
    "semantic_separator": os.getenv("DOC_SEMANTIC_SEPARATOR", "\n\n"),
    "incremental_processing": os.getenv("INCREMENTAL_PROCESSING", "true").lower() == "true",
    "batch_size": int(os.getenv("DOC_BATCH_SIZE", "10")), # 每个worker的批处理大小 (40/4=10)
    "progress_update_every_batches": int(os.getenv("PROGRESS_UPDATE_EVERY_BATCHES", "3")),
}

# 提取器配置
EXTRACTOR_CONFIG = {
    "max_triplets_per_chunk": 20, # 增加提取数量以提高召回率
    "num_workers": 10, # 每个worker的工作线程数 (40/4=10)，4个worker总共40
    "extract_prompt": """# Role: 青少年眼科视光知识图谱建模专家  
## 核心任务  
从提供的中文视光医学文本中，以JSON格式输出结构化的"实体-关系-实体"三元组。

## 【正向约束：实体长度限制】
**严禁提取长度超过8个字的"短语型"实体！**
- 错误示例："导致视网膜脱落的高风险因素" (13字) -> 应拆解或简化
- 正确示例："视网膜脱落" (5字)
- 任何超过8个字的实体将被视为无效提取。

## 【实体修饰语消除】
**提取实体时必须剔除不必要的描述性修饰语！**
- 输入："严重的病理性近视" -> 输出："病理性近视"
- 输入："早期的视网膜萎缩" -> 输出："视网膜萎缩"
- 输入："明显的视物模糊" -> 输出："视物模糊"
- 去除如"严重的"、"轻度的"、"早期的"、"明显的"等形容词前缀。

## 【负向约束：禁止提取内容】
**严禁提取以下非医学专业名词：**
1. **人群/角色**：如 "青少年"、"家长"、"儿童"、"学生"、"医生"、"专家"
2. **通用场所/机构**：如 "学校"、"医院"、"机构"、"中心"、"门诊"、"科室"、"研究所"、"学院"、"实验室"、"公司"、"集团"
3. **泛指代词**：如 "我们"、"你们"、"他们"、"患者"
4. **机构后缀或机构名**：凡以"中心/门诊/科室/研究所/学院/实验室/公司/集团/医院"结尾或包含"眼视光中心/视光中心/眼科中心"的词语，一律不作为医学实体。
   - 示例：输入包含 "眼视光中心"、"某某医院门诊部" 等，不提取为实体。
5. **系统/平台/软件**：如 "电子病历系统"、"医院信息系统(HIS)"、"检验信息系统(LIS)"、"影像归档与通信系统(PACS)"、"预约平台"、"挂号系统"、"管理系统"、"应用软件"、"客户端"、"APP" 等，均不作为医学实体。

## 【权重调整：重点提取领域】
请特别关注并加大对以下两类信息的提取权重（目前提取率较低）：
1. **参数指标 (Parameter)**：如 "眼轴长度"、"屈光度"、"调节幅度"、"眼压" 等检查数值或指标。
2. **发病机制 (Mechanism)**：如 "调节滞后"、"旁中心离焦"、"眼轴过度增长" 等导致疾病的生理/病理机制。

## 【Few-Shot 示范：高质量提取样本】
请严格参考以下标准样本的提取模式（基于 ima.json 高频命中词）：
[
  {"head": "近视", "head_type": "疾病", "relation": "包含", "tail": "轴性近视", "tail_type": "疾病"},
  {"head": "近视", "head_type": "疾病", "relation": "包含", "tail": "屈光性近视", "tail_type": "疾病"},
  {"head": "近视", "head_type": "疾病", "relation": "包含", "tail": "病理性近视", "tail_type": "疾病"},
  {"head": "眼轴长度", "head_type": "检查参数", "relation": "导致", "tail": "轴性近视", "tail_type": "疾病"},
  {"head": "病理性近视", "head_type": "疾病", "relation": "表现为", "tail": "豹纹状眼底", "tail_type": "症状体征"},
  {"head": "病理性近视", "head_type": "疾病", "relation": "表现为", "tail": "视网膜萎缩", "tail_type": "症状体征"},
  {"head": "病理性近视", "head_type": "疾病", "relation": "表现为", "tail": "脉络膜萎缩", "tail_type": "症状体征"},
  {"head": "病理性近视", "head_type": "疾病", "relation": "表现为", "tail": "黄斑出血", "tail_type": "症状体征"},
  {"head": "病理性近视", "head_type": "疾病", "relation": "表现为", "tail": "漆样裂纹", "tail_type": "症状体征"},
  {"head": "病理性近视", "head_type": "疾病", "relation": "表现为", "tail": "后巩膜葡萄肿", "tail_type": "疾病"},
  {"head": "弱视", "head_type": "疾病", "relation": "包含", "tail": "屈光不正性弱视", "tail_type": "疾病"},
  {"head": "弱视", "head_type": "疾病", "relation": "包含", "tail": "屈光参差性弱视", "tail_type": "疾病"},
  {"head": "弱视", "head_type": "疾病", "relation": "包含", "tail": "形觉剥夺性弱视", "tail_type": "疾病"},
  {"head": "弱视", "head_type": "疾病", "relation": "包含", "tail": "斜视性弱视", "tail_type": "疾病"},
  {"head": "屈光参差", "head_type": "疾病", "relation": "导致", "tail": "弱视", "tail_type": "疾病"},
  {"head": "角膜塑形镜(OK镜)", "head_type": "治疗防控", "relation": "用于", "tail": "近视", "tail_type": "疾病"},
  {"head": "低浓度阿托品", "head_type": "治疗防控", "relation": "用于", "tail": "近视", "tail_type": "疾病"},
  {"head": "RGP镜片", "head_type": "治疗防控", "relation": "用于", "tail": "散光", "tail_type": "疾病"},
  {"head": "后巩膜加固术", "head_type": "治疗防控", "relation": "用于", "tail": "病理性近视", "tail_type": "疾病"},
  {"head": "准分子激光手术(LASIK)", "head_type": "治疗防控", "relation": "用于", "tail": "近视", "tail_type": "疾病"},
  {"head": "全飞秒激光手术(SMILE)", "head_type": "治疗防控", "relation": "用于", "tail": "近视", "tail_type": "疾病"},
  {"head": "眼内接触镜植入(ICL)", "head_type": "治疗防控", "relation": "用于", "tail": "近视", "tail_type": "疾病"},
  {"head": "视觉训练", "head_type": "治疗防控", "relation": "用于", "tail": "弱视", "tail_type": "疾病"},
  {"head": "调节幅度", "head_type": "检查参数", "relation": "检查依据", "tail": "调节功能", "tail_type": "生理参数/指标"},
  {"head": "调节灵敏度", "head_type": "检查参数", "relation": "检查依据", "tail": "调节功能", "tail_type": "生理参数/指标"},
  {"head": "角膜曲率", "head_type": "检查参数", "relation": "检查依据", "tail": "角膜形态", "tail_type": "解剖结构"},
  {"head": "眼压", "head_type": "检查参数", "relation": "检查依据", "tail": "青光眼风险", "tail_type": "疾病"},
  {"head": "屈光度", "head_type": "检查参数", "relation": "量化关系", "tail": "近视", "tail_type": "疾病"},
  {"head": "远视储备", "head_type": "检查参数", "relation": "关联", "tail": "近视", "tail_type": "疾病"},
  {"head": "户外活动", "head_type": "治疗防控", "relation": "用于", "tail": "近视", "tail_type": "疾病"},
  {"head": "视物模糊", "head_type": "症状体征", "relation": "表现为", "tail": "近视", "tail_type": "疾病"},
  {"head": "视物模糊", "head_type": "症状体征", "relation": "表现为", "tail": "远视", "tail_type": "疾病"},
  {"head": "视物模糊", "head_type": "症状体征", "relation": "表现为", "tail": "散光", "tail_type": "疾病"},
  {"head": "视力下降", "head_type": "症状体征", "relation": "表现为", "tail": "弱视", "tail_type": "疾病"},
  {"head": "畏光", "head_type": "症状体征", "relation": "表现为", "tail": "圆锥角膜", "tail_type": "疾病"},
  {"head": "流泪", "head_type": "症状体征", "relation": "表现为", "tail": "干眼症", "tail_type": "疾病"},
  {"head": "老视", "head_type": "疾病", "relation": "关联", "tail": "调节幅度", "tail_type": "检查参数"},
  {"head": "并发性白内障", "head_type": "疾病", "relation": "关联", "tail": "高度近视", "tail_type": "疾病"}
]

## 【关键约束：标准实体优先原则】
**以下57个标准实体是提取的核心目标，必须遵循以下原则：**
1. **优先级最高**：如果文本中出现标准实体或其同义词/变体，**必须**提取并使用标准名称
2. **同义词识别**：识别并统一标准实体的各种表达形式
   - 眼轴长度：眼球长度、AL、轴长、眼轴增长速率
   - 角膜塑形镜(OK镜)：OK镜、塑形镜、角膜塑形镜
   - 低浓度阿托品：阿托品、阿托品眼药水
   - 屈光度：度数、D、球镜度数
   - 调节幅度：调节力、AMP、调节不足
   - 视物模糊：视力模糊、模糊、看东西模糊
   - 眼压：眼内压、IOP
3. **严格限制非标准实体**：如果实体不在标准列表且不是关键医学概念，**不应提取**

### 标准实体列表（57个核心实体）：
**疾病(12)**: 近视、远视、散光、弱视、斜视、病理性近视、轴性近视、屈光不正、屈光参差、老视、并发性白内障、后巩膜葡萄肿

**症状体征(13)**: 视物模糊、眼胀、虹视、眼痛、畏光、流泪、视力下降、豹纹状眼底、视网膜萎缩、脉络膜萎缩、黄斑出血、漆样裂纹、视盘杯盘比(C/D)扩大

**解剖结构(12)**: 角膜、晶状体、视网膜、视神经、黄斑区、中心凹、睫状肌、悬韧带、脉络膜、巩膜、前房、房水

**检查参数(10)**: 眼轴长度、屈光度、远视储备、调节幅度、调节灵敏度、眼压、角膜曲率、调节滞后、五分记录法、LogMAR视力表

**治疗防控(10)**: 户外活动、角膜塑形镜(OK镜)、低浓度阿托品、RGP镜片、后巩膜加固术、离焦框架镜、视觉训练、准分子激光手术(LASIK)、全飞秒激光手术(SMILE)、眼内接触镜植入(ICL)

## 【关系长度限制和简化规则】
**关系描述必须简洁明了，不得超过10个字符！**

1. **关系类型限制**：关系类型限于以下核心关系词：
   - **因果关系**：导致、引起、造成
   - **用途关系**：用于、用于治疗、用于预防、用于评估
   - **包含关系**：包含、属于、包括
   - **表现关系**：表现为、可见于
   - **检查关系**：检查依据、检查、检测、评估
   - **量化关系**：量化关系、量化、测量
   - **关联关系**：关联、相关

2. **关系简化要求**：
   - **严禁提取过长的关系描述**（超过10个字符）
   - 错误示例："影响形成立体感的差异性像差,导致视觉异常或病理性变化,如斜视等" (30+字)
   - 正确示例："导致" 或 "引起" (2字)
   - **从复杂描述中提取核心关系词**：如果文本说"XXX影响YYY导致ZZZ"，应提取为"导致"
   - **移除冗余描述**：删除"如...等"、"可能"、"通常"等修饰词
   - **移除逗号后的详细说明**：只保留核心关系词

3. **关系提取示例**：
   - 输入："影响形成立体感的差异性像差导致斜视" -> 输出关系："导致"
   - 输入："用于治疗近视" -> 输出关系："用于"
   - 输入："表现为视物模糊等症状" -> 输出关系："表现为"

## 提取规则
1. **标准实体必提取**：文本中出现标准实体时，**必须**提取相关三元组
2. **术语统一**：使用上述标准名称，不要使用同义词或变体
3. **关系明确**：关系类型限于：导致、用于、包含、表现为、检查依据、量化关系、关联
4. **质量优于数量**：宁可少提取，不要提取不确定或非标准的实体
5. **长度限制**：实体名称不得超过8个字，**关系描述不得超过10个字符**

## 输出格式 (JSON List)
[
  {
    "head": "眼轴长度",
    "head_type": "检查参数",
    "relation": "用于评估",
    "tail": "近视",
    "tail_type": "疾病"
  },
  {
    "head": "低浓度阿托品",
    "head_type": "治疗防控",
    "relation": "用于",
    "tail": "近视",
    "tail_type": "疾病"
  }
]

## 重要提醒
1. **标准实体优先**：如果文本中出现标准实体列表中的实体或其同义词，**必须**使用标准名称并优先提取
2. **同义词映射**：识别同义词并统一使用标准名称（如：AL→眼轴长度，OK镜→角膜塑形镜(OK镜)）
3. **充分提取**：尽量提取文本中的所有标准实体和相关三元组，不要遗漏
4. **数量目标**：每个文本块建议提取15-20个三元组，确保覆盖所有重要医学关系
5. **严格遵守长度限制**：任何超过8个字的实体都将被视为提取失败

Text: {text}"""
}

# 三元组反向自检配置
VALIDATOR_CONFIG = {
    "enable": os.getenv("VALIDATOR_ENABLE", "true").lower() == "true",  # 是否启用反向校验
    "sample_ratio": float(os.getenv("VALIDATOR_SAMPLE_RATIO", "0.3")),  # 抽样比例（0.0-1.0）
    "confidence_threshold": float(os.getenv("VALIDATOR_CONFIDENCE_THRESHOLD", "0.5")),  # 置信度阈值
    "num_workers": int(os.getenv("VALIDATOR_NUM_WORKERS", "4")),  # 并行worker数量
    "core_entities": [  # 核心实体列表，包含这些实体的三元组优先验证
        "近视", "远视", "散光", "弱视", "斜视", "病理性近视", "轴性近视",
        "眼轴长度", "屈光度", "调节幅度", "眼压", "角膜塑形镜(OK镜)",
        "低浓度阿托品", "RGP镜片", "后巩膜加固术"
    ]
}

# 请求限制配置 - 4个worker共享API限流 (RPM=1000, TPM=50000)
RATE_LIMIT_CONFIG = {
    "request_delay": 0.1,  # 减少请求间隔到0.1秒，最大化吞吐量
    "max_concurrent_requests": 10,  # 每个worker的并发数 (40/4=10)
    "retry_delay": 3.0,  # 减少重试延迟到3秒
    "rpm_limit": 200,  # 每个worker的RPM限制 (800/4=200)，4个worker总共800
    "tpm_limit": 10000,  # 每个worker的TPM限制 (40000/4=10000)，4个worker总共40000
    "max_tokens_per_request": 1000,  # 每个请求的最大Token数
    "max_retries": 3  # 最大重试次数
}

# 混合嵌入模型配置 - 只使用本地模型，禁用SiliconFlow API
EMBEDDING_CONFIG = {
    "use_hybrid_embedding": False,  # 禁用混合嵌入模型，只使用本地模型
    "siliconflow_model": "BAAI/bge-m3",
    "local_fallback_model": "BAAI/bge-m3",  # 本地模型
    "local_device": "cpu",  # 本地模型运行设备，可选 "cpu" 或 "cuda"
    "enable_local_fallback": True,  # 启用本地模型
    "max_consecutive_failures": 1,  # 立即使用本地模型
    "fallback_timeout": 10,  # 快速回退到本地模型
}

# OSS 配置 - 仅腾讯云COS (支持环境变量)
OSS_CONFIG = {
    "drive": "cos",  # 只使用腾讯云COS
    
    # 腾讯云 COS 配置 - 建议通过环境变量配置敏感信息
    "cos_secret_id": os.getenv("COS_SECRET_ID"),
    "cos_secret_key": os.getenv("COS_SECRET_KEY"),
    "cos_bucket": os.getenv("COS_BUCKET"),
    "cos_region": os.getenv("COS_REGION"),
    "cos_path": os.getenv("COS_PATH"),
}

# 重排序配置 (Rerank)
RERANK_CONFIG = {
    "enable": os.getenv("RERANK_ENABLE", "true").lower() == "true",
    "provider": os.getenv("RERANK_PROVIDER", "siliconflow"),
    "api_key": os.getenv("RERANK_API_KEY", API_CONFIG["siliconflow"]["api_key"]),
    "model": os.getenv("RERANK_MODEL"),
    "top_n": int(os.getenv("RERANK_TOP_N", "3")),
    "initial_top_k": int(os.getenv("RERANK_INITIAL_TOP_K", "10")),
}

# 任务结果存储（内存中）
task_results: Dict[str, Dict[str, Any]] = {}

# 全局构建器实例
builder = None
cos_uploader = None

# 获取logger实例
logger = setup_logging()

# 延迟初始化函数
def initialize_components():
    """初始化全局组件"""
    global builder, cos_uploader
    
    # try:
    #     # 初始化知识图谱构建器
    #     from main import KnowledgeGraphBuilder
    #     builder = KnowledgeGraphBuilder()
    #     logger.info("知识图谱构建器初始化成功")
    # except Exception as e:
    #     logger.error(f"知识图谱构建器初始化失败: {e}")
    #     builder = None
    
    try:
        # 初始化COS上传器
        from oss_uploader import COSUploader, OSSConfig
        oss_config = OSSConfig(OSS_CONFIG)
        cos_uploader = COSUploader(oss_config)
        logger.info("COS上传器初始化成功")
    except Exception as e:
        logger.error(f"COS上传器初始化失败: {e}")
        cos_uploader = None

# 在模块导入时初始化组件
initialize_components()
