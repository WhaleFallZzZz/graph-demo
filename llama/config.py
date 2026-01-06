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
                 可以通过环境变量 LOG_DIR 进行配置
    """
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
        "max_tokens": int(os.getenv("SILICONFLOW_MAX_TOKENS", "1000")),
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
    "num_workers": int(os.getenv("DOCUMENT_NUM_WORKERS", "1")),
    "chunk_size": int(os.getenv("DOC_CHUNK_SIZE", "1024")),
    "chunk_overlap": int(os.getenv("DOC_CHUNK_OVERLAP", "120")),
    "max_chunk_length": int(os.getenv("DOC_MAX_CHUNK_LENGTH", "1400")),
    "min_chunk_length": int(os.getenv("DOC_MIN_CHUNK_LENGTH", "600")),
    "dynamic_chunking": os.getenv("DOC_DYNAMIC_CHUNKING", "true").lower() == "true",
    "dynamic_target_chars_per_chunk": int(os.getenv("DOC_TARGET_CHARS_PER_CHUNK", "1200")),
    "benchmark_chunking": os.getenv("DOC_BENCHMARK_CHUNKING", "false").lower() == "true",
    "log_chunk_metrics": os.getenv("DOC_LOG_CHUNK_METRICS", "true").lower() == "true",
    "sentence_splitter": os.getenv("DOC_SENTENCE_SPLITTER", "。！？!?"),
    "semantic_separator": os.getenv("DOC_SEMANTIC_SEPARATOR", "\n\n"),
    "incremental_processing": os.getenv("INCREMENTAL_PROCESSING", "true").lower() == "true"
}

# 提取器配置
EXTRACTOR_CONFIG = {
    "max_triplets_per_chunk": 20, # 增加提取数量以提高召回率
    "num_workers":8,
    "extract_prompt": """# Role: 青少年眼科视光知识图谱建模专家  
## 核心任务  
从提供的中文视光医学文本中，以JSON格式输出结构化的"实体-关系-实体"三元组。**必须优先提取标准实体列表中的实体，非标准实体应严格限制提取。**

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

## 提取规则
1. **标准实体必提取**：文本中出现标准实体时，**必须**提取相关三元组
2. **术语统一**：使用上述标准名称，不要使用同义词或变体
3. **关系明确**：关系类型限于：导致、用于、包含、表现为、检查依据、量化关系、关联
4. **质量优于数量**：宁可少提取，不要提取不确定或非标准的实体

## 参考词库约束  
在提取实体时，可参考不限于以下分类：  
1. 屈光不正与生物参数类
    轴性近视 
    屈光性近视
    假性近视 
    高度近视 
    进行性近视 
    病理性近视 
    远视 
    规则散光 
    不规则散光 
    屈光参差 
    正视化 
    远视储备 
    眼轴长度 
    角膜曲率
    AL/CR比值 
    等效球镜 
    散光轴位 
    周边远视离焦
    晶状体屈光力
    前房深度 
2. 斜视、隐斜与双眼视觉类 
    调节性内斜视 
    非调节性内斜视
    间歇性外斜视
    恒定性外斜视 
    上斜视 
    旋转斜视
    内隐斜 
    外隐斜
    垂直隐斜
    微斜视
    眼球震颤
    异常视网膜对应 
    抑制 
    融合功能 
    立体视锐度 
    感觉融像 
    运动融像 
    复视
    注视性质 
    旁中心注视 
3. 调节与聚散功能异常类 
    调节不足
    调节过度 
    调节灵敏度下降
    调节滞后 
    调节超前 
    调节幅度
    集合不足 
    集合过度 
    散开不足 
    散开过度 
    AC/A比率 
    集合近点
    负相对调节
    正相对调节 
    调节性集合 
    融像性聚散
    负融像性聚散 
    正融像性聚散 
    扫视运动异常
    追随运动异常 
4. 弱视、眼表与防控产品类
    屈光不正性弱视 
    斜视性弱视 
    形觉剥夺性弱视 
    屈光参差性弱视 
    拥挤现象 
    干眼症 
    过敏性结膜炎 
    睑缘炎 
    视频终端综合征 
    视疲劳
    角膜塑形镜 
    硬性透氧性接触镜 
    离焦框架眼镜
    多焦点软性接触镜 
    低浓度阿托品 
    遮盖疗法 
    压抑疗法
    视觉训练
    海丁格刷
    视后像疗法 
5. 检查、病症与综合类 
    睫状肌麻痹验光
    综合验光仪检查
    角膜地形图 
    光学生物测量
    裂隙灯检查 
    值得四点试验
    马氏杆检查 
    立体视检查图
    先天性白内障 
    青少年型青光眼
    早产儿视网膜病变
    视网膜色素变性 
    色觉异常 
    夜盲症
    倒睫 
    上睑下垂 
    圆锥角膜
    视乳头水肿 
    巩膜加固术
    视觉发育关键期 

## 任务拆解  
### 1. 实体类型锁定  
- **类别定义**：实体必须根基于参考词库，具体类型包括：  
  - 眼部疾病/异常（如高度近视、散光轴位）。  
  - 生理参数/指标（如眼轴长度AL、调节幅度）。  
  - 检查项目（如光学生物测量、Sheard准则评估）。  
  - 治疗/防控手段（如角膜塑形镜、视觉训练）。  
  - 症状/体征（如漏字跳行、近视力模糊）。  
  - 评估标准（如Percival准则、1:1法则）。  

### 2. 关系类型定义  
- **关系类型**：仅限于以下六类：  
  - 导致关系（例如，“眼轴增长导致近视加深”）。  
  - 用于关系（例如，“睫状肌麻痹验光用于屈光矫正”）。  
  - 包含关系（例如，“轴性近视属于近视”）。  
  - 表现为关系（例如，“散光表现为垂直异位”）。  
  - 检查依据关系（例如，“眼轴长度是通过光学生物测量检查”）。  
  - 量化阈值关系（例如，“眼轴增长阈值为0.2mm/年”）。  

## 约束规则  
1. **强制标准实体优先**：文本中出现标准实体列表中的实体或其同义词时，**必须**提取并使用标准名称
2. **限制非标准实体**：非标准实体仅在确实重要且使用规范医学术语时才可提取，避免提取过多非关键实体
3. **术语一致性**：同一实体在整个输出中必须使用相同的标准名称
4. **拒绝低质量实体**：禁止提取非医学相关内容（如家庭成员、日常生活描述等）
5. **格式规范**：输出必须严格为JSON格式，每个三元组包含head、head_type、relation、tail、tail_type

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
