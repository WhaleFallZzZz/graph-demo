# 知识图谱构建系统

基于LlamaIndex、SiliconFlow API和Neo4j的知识图谱构建系统。

## 系统架构

### 核心模块

- **`config.py`** - 系统配置管理
  - API配置（SiliconFlow）
  - Neo4j数据库配置
  - 文档处理配置
  - 频率限制配置

- **`utils.py`** - 工具函数模块
  - JSON文本清理和修复
  - 三元组提取和解析
  - 正则表达式备用解析

- **`custom_siliconflow_llm.py`** - 自定义LLM实现
  - 流式请求支持，避免超时
  - 智能频率控制
  - 详细的错误处理和统计

- **`main.py`** - 主程序入口
  - 知识图谱构建器类
  - 模块化组件管理
  - 完整的错误处理

### 功能特性

✅ **嵌入模型**：SiliconFlow BAAI/bge-m3 API  
✅ **语言模型**：SiliconFlow Qwen/Qwen2.5-7B-Instruct  
✅ **图数据库**：Neo4j  
✅ **文档支持**：txt, docx, pdf  
✅ **频率控制**：智能请求间隔  
✅ **错误处理**：完善的异常处理机制  
✅ **日志系统**：详细的运行日志  

## 快速开始

### 1. 环境要求

```bash
pip install llama-index llama-index-llms-siliconflow llama-index-embeddings-siliconflow llama-index-graph-stores-neo4j neo4j aiohttp python-dotenv pypdf
```

> **注意**: `pypdf` 用于解析 PDF 文件，如果只处理 txt 和 docx 文件可以不安装。

### 2. 配置环境变量（推荐）

创建 `.env` 文件配置敏感信息：

```bash
# LLM配置
SILICONFLOW_API_KEY=your_api_key_here
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
EMBEDDING_MODEL=BAAI/bge-m3

# Neo4j配置
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password_here
NEO4J_URL=bolt://localhost:7687
NEO4J_DATABASE=neo4j

# 腾讯云COS配置
COS_SECRET_ID=your_secret_id_here
COS_SECRET_KEY=your_secret_key_here
COS_BUCKET=your_bucket_name
COS_REGION=ap-shanghai

# 文档路径
DOCUMENT_PATH=/path/to/your/documents

# 日志目录 (可选，默认为项目根目录下的 logs)
LOG_DIR=/path/to/your/custom/logs

```

在主程序开始处添加：
```python
from dotenv import load_dotenv
load_dotenv()  # 加载 .env 文件
```

### 3. 配置Neo4j

确保Neo4j服务正在运行：
- 地址：bolt://localhost:7687
- 用户名：neo4j
- 密码：12345678

### 4. 准备文档

将文档放入配置的文档目录（默认 `/Users/whalefall/Downloads/neo4j-data`），支持：
- `.txt` 文件
- `.docx` 文件
- `.pdf` 文件
- `.html` / `.htm` 文件
- `.md` 文件
- 代码文件（`.py`, `.js`, `.json`, `.yaml` 等）

### 5. 运行程序

```bash
cd /Users/whalefall/Documents/workspace/python_demo/llama
python main.py
```

## 配置说明

### 环境变量配置（推荐方式）

使用 `.env` 文件配置所有敏感信息，支持的环境变量：

**LLM配置**
- `SILICONFLOW_API_KEY`: SiliconFlow API密钥
- `LLM_MODEL`: 大语言模型名称（默认：Qwen/Qwen2.5-7B-Instruct）
- `EMBEDDING_MODEL`: 嵌入模型名称（默认：BAAI/bge-m3）
- `LLM_TIMEOUT`: 请求超时时间（默认：120秒）
- `MAX_TOKENS`: 最大输出长度（默认：500）
- `LLM_TEMPERATURE`: 温度参数（默认：0.0）

**Neo4j配置**
- `NEO4J_USERNAME`: Neo4j用户名（默认：neo4j）
- `NEO4J_PASSWORD`: Neo4j密码
- `NEO4J_URL`: Neo4j连接URL（默认：bolt://localhost:7687）
- `NEO4J_DATABASE`: Neo4j数据库名（默认：neo4j）

**腾讯云COS配置**
- `COS_SECRET_ID`: 腾讯云SecretId
- `COS_SECRET_KEY`: 腾讯云SecretKey
- `COS_BUCKET`: COS存储桶名称
- `COS_REGION`: COS地域（默认：ap-shanghai）
- `COS_PATH`: COS存储路径（默认：/upload/neo4j）

**其他配置**
- `DOCUMENT_PATH`: 文档目录路径
- `LOG_DIR`: 日志目录路径
- `MAX_PATHS_PER_CHUNK`: 每块最大三元组数量（默认：3）
- `NUM_WORKERS`: 工作线程数（默认：1）

### 代码配置（备选方式）

如果不使用环境变量，可以直接在 `config.py` 中修改配置

## 错误处理

系统包含多层错误处理：
1. **网络层**：HTTP请求重试和超时控制
2. **API层**：频率限制和流式响应
3. **解析层**：JSON修复和正则表达式备用
4. **应用层**：完整的异常捕获和日志记录

## 性能优化

- **流式请求**：避免长文本超时
- **频率控制**：防止API限流
- **模块化设计**：清晰的代码结构
- **延迟导入**：加快启动速度

## 日志查看

运行日志保存在：`/Users/whalefall/Documents/workspace/python_demo/logs/llama_index_YYYY-MM-DD.log`
（例如：`llama_index_2025-12-24.log`）

## 故障排除

### 403 Forbidden错误
- 检查API密钥是否正确
- 确认账户有流式请求权限
- 降低请求频率

### 超时错误
- 流式请求已启用，通常不会超时
- 检查网络连接稳定性
- 减小max_tokens值

### Neo4j连接失败
- 确认Neo4j服务正在运行
- 检查连接参数配置
- 验证用户名和密码