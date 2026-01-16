# 知识图谱构建系统 - 代码架构文档

## 目录
- [系统概述](#系统概述)
- [代码层级结构](#代码层级结构)
- [核心模块详解](#核心模块详解)
- [数据流程](#数据流程)
- [配置说明](#配置说明)
- [使用场景](#使用场景)

---

## 系统概述

本系统是一个基于 LlamaIndex、SiliconFlow API 和 Neo4j 的知识图谱构建平台，主要功能包括：

1. **文档处理**：支持多种格式文档的解析和分块
2. **实体提取**：从文档中提取实体和关系
3. **知识图谱构建**：将提取的三元组存储到 Neo4j 图数据库
4. **图查询服务**：提供基于知识图谱的问答和检索服务
5. **实时进度追踪**：通过 SSE 提供实时进度反馈

---

## 代码层级结构

```
python_demo/
├── llama/                          # 核心模块目录
│   ├── config.py                   # 系统配置管理
│   ├── factories.py                # 工厂模式 - 组件创建
│   ├── kg_manager.py              # 知识图谱构建管理器
│   ├── graph_service.py           # 图服务 - 业务逻辑层
│   ├── graph_agent.py             # 图代理 - 查询和推理
│   ├── server.py                  # Flask 服务器 - API 接口层
│   │
│   ├── entity_extractor.py        # 实体提取器 - 核心提取逻辑
│   ├── enhanced_entity_extractor.py # 增强实体提取器 - 标准化映射
│   │
│   ├── semantic_chunker.py        # 语义分块器
│   ├── improved_semantic_chunker.py # 改进的语义分块器
│   │
│   ├── custom_siliconflow_embedding.py # 自定义嵌入模型
│   ├── custom_siliconflow_rerank.py   # 自定义重排序模型
│   │
│   ├── neo4j_text_sanitizer.py   # Neo4j 文本清理器
│   ├── neo4j_batch_ops.py       # Neo4j 批量操作
│   │
│   ├── ocr_parser.py            # OCR 解析器（PDF）
│   ├── file_type_detector.py    # 文件类型检测
│   ├── oss_uploader.py         # 腾讯云 COS 上传器
│   │
│   ├── progress_sse.py          # SSE 进度推送
│   ├── dynamic_concurrency_manager.py # 动态并发管理
│   ├── dynamic_resource_allocator.py  # 动态资源分配
│   │
│   ├── relation_normalizer.py   # 关系规范化
│   ├── entity_type_rules.py    # 实体类型规则
│   ├── query_intent.py         # 查询意图识别
│   ├── llm_intent_classifier.py # LLM 意图分类
│   │
│   ├── graph_context_postprocessor.py    # 图上下文后处理
│   ├── semantic_enrichment_postprocessor.py # 语义增强后处理
│   │
│   └── common/                 # 通用工具模块
│       ├── json_utils.py       # JSON 工具
│       ├── text_utils.py       # 文本工具
│       ├── file_utils.py       # 文件工具
│       ├── cache_utils.py      # 缓存工具
│       ├── concurrent_utils.py  # 并发工具
│       ├── error_handler.py    # 错误处理
│       ├── config_manager.py   # 配置管理
│       ├── datetime_utils.py    # 日期时间工具
│       └── neo4j_batch_ops.py # Neo4j 批量操作
│
├── scripts/                    # 脚本工具目录
│   ├── offline_entity_alignment.py  # 离线实体对齐
│   └── export_neo4j_triplets.py    # Neo4j 数据导出
│
└── logs/                       # 日志目录
```

---

## 核心模块详解

### 1. 配置层 (Configuration Layer)

#### `config.py` - 系统配置管理
**层级**：最底层配置中心

**主要功能**：
- 集中管理所有系统配置
- 支持环境变量和代码配置
- 初始化全局组件（COS 上传器等）
- 提供日志设置

**配置项**：
- `API_CONFIG`: SiliconFlow API 配置（LLM、Embedding 模型）
- `NEO4J_CONFIG`: Neo4j 数据库连接配置
- `EMBEDDING_CONFIG`: 嵌入模型配置
- `DOCUMENT_CONFIG`: 文档处理配置
- `EXTRACTOR_CONFIG`: 实体提取配置
- `RATE_LIMIT_CONFIG`: 频率限制配置
- `OSS_CONFIG`: 腾讯云 COS 配置

**实际用途**：
- 统一配置入口，避免配置分散
- 支持环境变量覆盖，便于部署
- 组件初始化，确保系统启动时所有必要组件就绪

---

### 2. 工厂层 (Factory Layer)

#### `factories.py` - 组件工厂
**层级**：组件创建层

**主要功能**：
- 延迟加载 LlamaIndex 模块
- 创建 LLM、Embedding、GraphStore 等核心组件
- 统一组件创建接口

**工厂类**：
- `LlamaModuleFactory`: 管理 LlamaIndex 模块导入
- `ModelFactory`: 创建 LLM 和 Embedding 模型
- `GraphStoreFactory`: 创建图存储（Neo4j 或内存）
- `ExtractorFactory`: 创建实体提取器
- `RerankerFactory`: 创建重排序模型

**实际用途**：
- 解耦组件创建和使用
- 延迟加载，加快启动速度
- 统一组件管理，便于替换实现

---

### 3. 知识图谱管理层 (Knowledge Graph Management Layer)

#### `kg_manager.py` - 知识图谱构建管理器
**层级**：核心业务逻辑层

**主要功能**：
- 文档加载和解析
- 文档分块（语义分块）
- 实体和关系提取
- 三元组存储到 Neo4j
- 进度追踪和错误处理

**核心类**：
- `DocumentIndex`: 文档倒排索引
- `KnowledgeGraphBuilder`: 知识图谱构建器

**实际用途**：
- 协调整个知识图谱构建流程
- 管理文档处理管道
- 处理批量操作和并发控制
- 提供进度反馈机制

**数据流程**：
```
文档 → 解析 → 分块 → 实体提取 → Neo4j 存储
```

---

### 4. 服务层 (Service Layer)

#### `graph_service.py` - 图服务
**层级**：业务服务层

**主要功能**：
- 提供知识图谱查询接口
- 管理构建任务
- 动态资源分配
- 并发控制

**核心类**：
- `GraphService`: 知识图谱服务

**实际用途**：
- 封装知识图谱操作的业务逻辑
- 提供统一的查询接口
- 管理任务队列和并发
- 实现动态资源调度

#### `graph_agent.py` - 图代理
**层级**：查询和推理层

**主要功能**：
- 基于知识图谱的问答
- 图遍历和推理
- 上下文检索
- 意图识别

**核心类**：
- `GraphAgent`: 图代理

**实际用途**：
- 实现基于知识图谱的智能问答
- 执行复杂的图查询和推理
- 提供上下文感知的检索

---

### 5. API 层 (API Layer)

#### `server.py` - Flask 服务器
**层级**：最顶层 API 接口

**主要功能**：
- 提供 RESTful API 接口
- 文件上传处理
- SSE 实时进度推送
- 错误处理和响应

**主要接口**：
- `POST /upload`: 文件上传
- `POST /build`: 构建知识图谱
- `GET /progress`: 获取构建进度（SSE）
- `POST /query`: 知识图谱查询

**实际用途**：
- 对外提供 HTTP API
- 支持前端集成
- 实现实时进度反馈
- 处理文件上传和下载

---

### 6. 实体提取层 (Entity Extraction Layer)

#### `entity_extractor.py` - 实体提取器
**层级**：核心算法层

**主要功能**：
- 从文本中提取实体和关系
- 三元组生成和验证
- JSON 解析和修复
- 批量处理

**核心类**：
- `EnhancedEntityExtractor`: 增强的实体提取器

**实际用途**：
- 实现实体识别算法
- 生成高质量的三元组
- 处理 JSON 解析错误
- 支持批量提取

#### `enhanced_entity_extractor.py` - 增强实体提取器
**层级**：增强算法层

**主要功能**：
- 实体标准化映射
- 同义词处理
- 向量相似度匹配（已注释）
- LRU 缓存优化

**核心类**：
- `StandardTermMapper`: 标准术语映射器

**实际用途**：
- 统一实体命名
- 处理同义词和变体
- 提高实体识别准确率
- 优化性能（缓存）

---

### 7. 文档处理层 (Document Processing Layer)

#### `semantic_chunker.py` - 语义分块器
**层级**：文档处理层

**主要功能**：
- 基于语义相似度的文档分块
- 句子向量计算
- 相似度阈值控制

**实际用途**：
- 将长文档分割成语义连贯的块
- 提高实体提取的准确性
- 支持自定义分块参数

#### `improved_semantic_chunker.py` - 改进的语义分块器
**层级**：增强文档处理层

**主要功能**：
- 改进的语义分块算法
- 更好的边界检测
- 优化分块质量

**实际用途**：
- 提供更精确的分块结果
- 减少语义断裂
- 提高知识图谱质量

---

### 8. 数据清理层 (Data Sanitization Layer)

#### `neo4j_text_sanitizer.py` - Neo4j 文本清理器
**层级**：数据清理层

**主要功能**：
- 清理实体名称（防注入）
- 清理关系标签
- 清理实体类型
- 特殊字符处理

**核心类**：
- `Neo4jTextSanitizer`: 文本清理器

**实际用途**：
- 防止 SQL/Cypher 注入攻击
- 确保数据符合 Neo4j 规范
- 清理无效字符
- 保留必要信息（防注入）

**已注释的功能**（按用户要求）：
- 标准实体列表映射
- 特殊字符转换
- 长度限制
- 关系规范化

---

### 9. 工具层 (Utility Layer)

#### `custom_siliconflow_embedding.py` - 自定义嵌入模型
**层级**：模型封装层

**主要功能**：
- 封装 SiliconFlow Embedding API
- 批量嵌入计算
- 频率限制和重试
- 错误处理

**实际用途**：
- 提供稳定的嵌入服务
- 支持批量计算
- 智能频率控制
- 完善的错误恢复

#### `custom_siliconflow_rerank.py` - 自定义重排序模型
**层级**：模型封装层

**主要功能**：
- 封装 SiliconFlow Rerank API
- 结果重排序
- 相关性评分

**实际用途**：
- 提高检索结果质量
- 优化排序准确性
- 支持批量重排序

#### `ocr_parser.py` - OCR 解析器
**层级**：文档解析层

**主要功能**：
- PDF 文件 OCR 解析
- 图像转文本
- DeepSeek-OCR 集成

**实际用途**：
- 处理扫描版 PDF
- 提取图片中的文字
- 支持多页文档

#### `file_type_detector.py` - 文件类型检测
**层级**：工具层

**主要功能**：
- 自动检测文件类型
- 支持多种格式
- MIME 类型识别

**实际用途**：
- 自动选择解析器
- 支持多种文件格式
- 提高系统兼容性

#### `oss_uploader.py` - 腾讯云 COS 上传器
**层级**：存储层

**主要功能**：
- 文件上传到腾讯云 COS
- 文件下载
- URL 生成

**实际用途**：
- 云存储集成
- 文件共享
- 备份和恢复

---

### 10. 进度追踪层 (Progress Tracking Layer)

#### `progress_sse.py` - SSE 进度推送
**层级**：通信层

**主要功能**：
- Server-Sent Events 实现
- 实时进度推送
- 多客户端支持
- 进度事件管理

**核心类**：
- `ProgressTracker`: 进度追踪器
- `ProgressManager`: 进度管理器

**实际用途**：
- 实时反馈构建进度
- 支持前端进度条
- 多任务并发追踪

---

### 11. 并发和资源管理层 (Concurrency and Resource Management Layer)

#### `dynamic_concurrency_manager.py` - 动态并发管理器
**层级**：并发控制层

**主要功能**：
- 动态调整并发数
- 负载均衡
- 资源优化

**实际用途**：
- 最大化系统吞吐量
- 避免资源过载
- 自适应性能优化

#### `dynamic_resource_allocator.py` - 动态资源分配器
**层级**：资源管理层

**主要功能**：
- 动态资源分配
- Worker 管理
- 资源监控

**实际用途**：
- 智能资源调度
- 多 Worker 协同
- 负载均衡

---

### 12. 后处理层 (Post-Processing Layer)

#### `graph_context_postprocessor.py` - 图上下文后处理
**层级**：数据增强层

**主要功能**：
- 图上下文增强
- 关系补充
- 实体属性完善

**实际用途**：
- 提高知识图谱完整性
- 补充隐含关系
- 丰富实体信息

#### `semantic_enrichment_postprocessor.py` - 语义增强后处理
**层级**：数据增强层

**主要功能**：
- 语义相似度计算
- 实体关联增强
- 关系类型优化

**实际用途**：
- 提高知识图谱质量
- 发现隐含关联
- 优化关系表达

---

### 13. 通用工具模块 (Common Utilities)

#### `common/json_utils.py` - JSON 工具
**主要功能**：
- JSON 解析和修复
- 错误处理
- 第三方库集成（json5）

**实际用途**：
- 处理 LLM 输出的 JSON
- 自动修复格式错误
- 提高解析成功率

#### `common/text_utils.py` - 文本工具
**主要功能**：
- 文本清理
- 字符串处理
- 正则表达式工具

**实际用途**：
- 文本预处理
- 格式统一
- 特殊字符处理

#### `common/cache_utils.py` - 缓存工具
**主要功能**：
- LRU 缓存
- 缓存管理
- 性能优化

**实际用途**：
- 减少重复计算
- 提高响应速度
- 降低 API 调用

#### `common/concurrent_utils.py` - 并发工具
**主要功能**：
- 线程池管理
- 异步任务
- 并发控制

**实际用途**：
- 提高处理效率
- 并发任务管理
- 资源优化

---

## 数据流程

### 1. 知识图谱构建流程

```
┌─────────────┐
│  文档上传   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  文件类型   │
│   检测      │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  文档解析   │
│ (OCR/TXT)   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  语义分块   │
│ (两阶段)    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  实体提取   │
│ (LLM)       │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  三元组     │
│   验证      │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  文本清理   │
│ (防注入)    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Neo4j     │
│   存储      │
└─────────────┘
```

### 2. 知识图谱查询流程

```
┌─────────────┐
│  用户查询   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  意图识别   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  查询解析   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  图遍历     │
│ (Neo4j)     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  结果重排   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  上下文     │
│   增强      │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  返回结果   │
└─────────────┘
```

---

## 配置说明

### 环境变量配置

```bash
# LLM 配置
SILICONFLOW_API_KEY=your_api_key
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
EMBEDDING_MODEL=BAAI/bge-m3
LLM_TIMEOUT=120
MAX_TOKENS=8192
LLM_TEMPERATURE=0.1

# Neo4j 配置
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_URL=bolt://localhost:7687
NEO4J_DATABASE=neo4j

# 腾讯云 COS 配置
COS_SECRET_ID=your_secret_id
COS_SECRET_KEY=your_secret_key
COS_BUCKET=your_bucket
COS_REGION=ap-shanghai

# 文档配置
DOCUMENT_PATH=/path/to/documents
CHUNK_SIZE=600
CHUNK_OVERLAP=100

# 提取配置
MAX_TRIPLETS_PER_CHUNK=120
MIN_ENTITIES_PER_CHUNK=20
ENTITY_CONFIDENCE_THRESHOLD=0.6
```

---

## 使用场景

### 1. 构建知识图谱

```python
from llama.kg_manager import builder

# 初始化构建器
builder.initialize()

# 构建知识图谱
builder.build_graph(
    document_path="/path/to/documents",
    use_ocr=True
)
```

### 2. 查询知识图谱

```python
from llama.graph_agent import GraphAgent

# 创建图代理
agent = GraphAgent()

# 查询实体
results = agent.query_entity("近视")

# 查询关系
results = agent.query_relation("导致", "近视", "弱视")
```

### 3. 导出数据

```bash
# 导出节点和边
python3 scripts/export_neo4j_triplets.py

# 离线实体对齐
python3 scripts/offline_entity_alignment.py
```

### 4. 启动 API 服务

```bash
# 启动 Flask 服务器
cd llama
python server.py
```

---

## 技术栈

- **LLM**: SiliconFlow (Qwen/Qwen2.5-7B-Instruct)
- **Embedding**: SiliconFlow (BAAI/bge-m3)
- **Graph Database**: Neo4j
- **Web Framework**: Flask
- **Document Processing**: PyMuPDF, python-docx
- **OCR**: DeepSeek-OCR
- **Storage**: 腾讯云 COS
- **Real-time Communication**: Server-Sent Events (SSE)

---

## 性能优化

1. **流式请求**: 避免长文本超时
2. **批量操作**: 减少 API 调用次数
3. **缓存机制**: LRU 缓存减少重复计算
4. **并发控制**: 动态调整并发数
5. **延迟加载**: 按需加载模块

---

## 错误处理

系统包含多层错误处理：

1. **网络层**: HTTP 请求重试和超时控制
2. **API 层**: 频率限制和流式响应
3. **解析层**: JSON 修复和正则表达式备用
4. **应用层**: 完整的异常捕获和日志记录

---

## 日志系统

日志文件位置：`logs/llama_index_YYYY-MM-DD.log`

日志级别：
- `INFO`: 正常运行信息
- `WARNING`: 警告信息
- `ERROR`: 错误信息
- `DEBUG`: 调试信息

---

## 扩展性

系统设计支持以下扩展：

1. **新的文档格式**: 添加新的解析器
2. **新的 LLM 模型**: 通过工厂模式替换
3. **新的图数据库**: 实现统一的 GraphStore 接口
4. **新的后处理器**: 添加到后处理管道
5. **新的提取策略**: 实现新的 Extractor

---

## 维护建议

1. **定期备份**: 备份 Neo4j 数据库
2. **监控日志**: 关注错误和警告信息
3. **性能调优**: 根据实际负载调整参数
4. **更新依赖**: 定期更新 Python 依赖
5. **清理缓存**: 定期清理过期缓存

---

## 联系方式

如有问题或建议，请联系开发团队。
