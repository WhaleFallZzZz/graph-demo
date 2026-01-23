# 眼视光知识图谱项目 (Ophthalmology Knowledge Graph)

本项目是一个专注于眼视光领域的知识图谱构建与查询系统。利用 **LlamaIndex**、**SiliconFlow API** (Qwen/BGE) 和 **Neo4j** 图数据库，实现从医疗文档（PDF/Docx/Txt）到知识图谱的自动化转化，并提供智能问答服务。

## 🌟 核心特性

- **多模态文档解析**：支持通过 OCR (DeepSeek-OCR) 处理扫描版 PDF、纯文本 PDF、Word 及 Txt 文件。
- **语义化分块 (Semantic Chunking)**：基于句子相似度进行文档切分，确保三元组提取的上下文完整性。
- **增强型实体提取**：集成 LLM 进行实体和关系提取，支持实体标准化映射与同义词处理。
- **动态图谱查询**：提供基于意图识别的图代理 (Graph Agent)，支持复杂的图遍历与语义增强检索。
- **实时进度追踪**：基于 SSE (Server-Sent Events) 的任务进度反馈，支持大规模文档构建。

---

## 📂 模块结构说明

核心代码位于 `llama/` 目录下，各模块分工如下：

| 模块名称 | 功能描述 |
| :--- | :--- |
| **`server.py`** | **API 层**。基于 Flask 提供 RESTful 接口（上传、构建、查询、SSE 进度）。 |
| **`graph_service.py`** | **服务层**。封装图谱构建任务管理、动态资源分配及并发控制逻辑。 |
| **`graph_agent.py`** | **图代理/推理层**。实现基于图数据库的智能问答、意图识别与上下文检索。 |
| **`kg_manager.py`** | **构建管理器**。协调解析、分块、提取及 Neo4j 存储的全流程管道。 |
| **`entity_extractor.py`** | **提取引擎**。处理 LLM 交互，从文本中提取三元组，包含 JSON 修复机制。 |
| **`enhanced_entity_extractor.py`** | **标准化映射**。负责实体术语的标准映射、同义词对齐与缓存优化。 |
| **`semantic_chunker.py`** | **语义分块器**。计算文本向量并基于相似度边界进行自适应分块。 |
| **`ocr_parser.py`** | **PDF 解析器**。集成 OCR 技术处理非文本格式的医疗报告或书籍。 |
| **`config.py` / `factories.py`** | **配置与工厂**。管理全局配置，通过工厂模式延迟加载组件以优化启动速度。 |
| **`common/`** | **工具库**。包含 JSON 修复、并发处理、缓存管理等通用辅助功能。 |

---

## 🚀 快速开始

### 1. 环境准备
项目要求 Python 3.8+，建议使用虚拟环境。

```bash
pip install -r llama/requirements.txt
```

### 2. 配置环境变量
在根目录下创建 `.env` 文件，完善以下核心配置：

```bash
# SiliconFlow API (LLM/Embedding)
SILICONFLOW_API_KEY=sk-xxxx
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
EMBEDDING_MODEL=BAAI/bge-m3

# Neo4j 数据库
NEO4J_URL=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# 腾讯云 COS (用于解析过程中的资源托管)
COS_SECRET_ID=xxx
COS_SECRET_KEY=xxx
COS_BUCKET=xxx
COS_REGION=ap-shanghai
```

### 3. 启动 API 服务
使用项目提供的启动脚本：

```bash
# 给脚本添加执行权限
chmod +x start_server.sh

# 启动服务 (默认监听 8001 端口)
./start_server.sh
```
*注：脚本会自动检测并优先使用 Gunicorn (多线程模式) 启动生产环境。*

---

## 🛠️ 常用脚本工具

位于 `scripts/` 目录：
- `offline_entity_alignment.py`: 离线对齐已有图谱中的同义实体。
- `export_neo4j_triplets.py`: 将 Neo4j 中的三元组导出为 JSON/CSV 格式。

---

## 🔍 技术栈
- **LLM/Embedding**: SiliconFlow (Qwen 2.5 / BGE-M3)
- **Knowledge Graph**: Neo4j / LlamaIndex KnowledgeGraphIndex
- **Backend**: Flask + Gunicorn + SSE
- **OCR**: DeepSeek-OCR
- **Storage**: Tencent Cloud COS
