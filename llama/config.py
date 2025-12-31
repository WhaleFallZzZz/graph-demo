"""
知识图谱构建配置模块
支持通过环境变量覆盖默认配置
"""

import os
import logging
from datetime import datetime
from typing import Dict, Any
from pathlib import Path

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 日志配置
def setup_logging(log_dir: str = None) -> logging.Logger:
    """设置日志配置，按日期生成日志文件
    
    Args:
        log_dir: 日志目录路径，默认为项目根目录下的 logs 文件夹
    """
    if log_dir is None:
        log_dir = os.getenv("LOG_DIR", str(PROJECT_ROOT / "logs"))
    """设置日志配置，按日期生成日志文件"""
    # 确保日志目录存在
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    # 生成带日期的日志文件名
    current_date = datetime.now().strftime("%Y-%m-%d")
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
    return logging.getLogger(__name__)

# API配置 - 支持通过环境变量配置
API_CONFIG = {
    "siliconflow": {
        "api_key": os.getenv("SILICONFLOW_API_KEY"),
        "llm_model": os.getenv("SILICONFLOW_LLM_MODEL"),
        "embedding_model": os.getenv("SILICONFLOW_EMBEDDING_MODEL"),
        "timeout": int(os.getenv("SILICONFLOW_TIMEOUT", "120")),
        "max_tokens": int(os.getenv("SILICONFLOW_MAX_TOKENS", "500")),
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
    "num_workers": int(os.getenv("DOCUMENT_NUM_WORKERS", "1"))
}

# 提取器配置
EXTRACTOR_CONFIG = {
    "max_triplets_per_chunk": 8, # 用于 DynamicLLMPathExtractor
    "extract_prompt": """你是青少年眼科视光领域的知识图谱提取专家，核心任务是从给定的青少年眼科视光相关文本中，精准提取符合语义逻辑的实体-关系-实体三元组，为视光领域知识体系构建提供结构化支撑。
【质量优先原则】
优先提取核心、重要、具有明确语义关系的三元组。宁可少提取，也要要确保每个三元组都是高质量、有价值、有意义的。避免提取冗余、重复、或语义模糊的三元组。
任务拆解（按以下步骤执行）
1. 文本范围锁定：仅处理与青少年眼科视光直接相关的文本（如近视防控、配镜、视功能训练、眼科检查等内容），若文本涉及非青少年群体或非视光领域信息，自动过滤该部分内容。
2. 实体识别与类型标注：
   - 对文本中的实体进行逐一识别，实体类型需基于语义分析判定为青少年眼科视光领域的具体分类（如：眼科检查项目、视光产品、眼部疾病、防控方法、医疗机构、专业人员等），禁止使用“实体”等泛指类型。
   - 若实体为英文，需先翻译为简体中文后再标注类型（如将“myopia”翻译为“近视”，类型标注为“眼部疾病”）。
3. 关系类型定义：关系类型需为青少年眼科视光领域的具体描述（如：导致、用于治疗、包含、属于、推荐年龄、检查依据、适配人群等），禁止使用模糊或泛指的关系描述。
4. 三元组构建：将识别的实体与关系组合为“头实体-关系-尾实体”三元组，确保逻辑连贯、符合文本语义。
5. 格式校验：输出结果必须严格遵循指定的JSON列表格式，确保无语法错误。

约束规则
● 必做：
  1. 实体类型和关系类型均需使用简体中文。
  2. 每个三元组的头实体、尾实体必须标注具体类型。
  3. 过滤非青少年眼科视光领域的信息。
● 禁止：
  1. 使用“实体”“关系”等泛指的类型描述。
  2. 保留未翻译的英文实体或关系。
  3. 构建逻辑冲突或不符合文本语义的三元组。

输出格式
必须为有效的JSON列表，示例如下：
[
  {
    "head": "角膜塑形镜",
    "head_type": "视光产品",
    "relation": "用于治疗",
    "tail": "青少年近视",
    "tail_type": "眼部疾病"
  },
  {
    "head": "散瞳验光",
    "head_type": "眼科检查项目",
    "relation": "推荐年龄",
    "tail": "6-12岁青少年",
    "tail_type": "人群"
  }
]
【重要提醒】只提取高质量、核心的三元组。如果文本中没有足够高质量的三元组，可以输出空列表[]。质量比数量更重要。
Text: {text}"""
}

# 请求限制配置 - 更严格的速率限制
RATE_LIMIT_CONFIG = {
    "request_delay": 0.5,  # 增加请求间隔到3秒，避免触发403错误
    "max_concurrent_requests": 5,  # 严格控制并发数
    "retry_delay": 15.0,  # 增加重试延迟到15秒
    "rpm_limit": 20,  # 大幅降低每分钟请求数限制
    "tpm_limit": 10000,  # 降低每分钟Token数限制
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