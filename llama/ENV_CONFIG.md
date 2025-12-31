# 环境变量配置说明

## 概述
本项目支持通过环境变量配置所有敏感信息和关键参数，提高安全性和灵活性。

## 配置方式

### 方式一：使用 .env 文件（推荐）

1. 在项目根目录创建 `.env` 文件
2. 复制下面的配置模板并填入实际值
3. 在主程序开始处添加：
```python
from dotenv import load_dotenv
load_dotenv()
```

### 方式二：直接设置环境变量

```bash
export SILICONFLOW_API_KEY=your_api_key_here
export NEO4J_PASSWORD=your_password_here
# ... 其他配置
```

## 配置模板

创建 `.env` 文件，内容如下：

```bash
# ==================== LLM配置 ====================
SILICONFLOW_API_KEY=your_api_key_here
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
EMBEDDING_MODEL=BAAI/bge-m3
LLM_TIMEOUT=120
MAX_TOKENS=500
MAX_RETRIES=3
LLM_TEMPERATURE=0.0

# ==================== Neo4j配置 ====================
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password_here
NEO4J_URL=bolt://localhost:7687
NEO4J_DATABASE=neo4j

# ==================== 文档处理配置 ====================
DOCUMENT_PATH=/path/to/your/documents
MAX_PATHS_PER_CHUNK=3
NUM_WORKERS=1

# ==================== 重排序配置（可选）====================
# 注意：需要在 config.py 中设置 RERANK_CONFIG['enable'] = True 才会启用
RERANK_API_KEY=your_rerank_api_key_here

# ==================== 腾讯云COS配置 ====================
COS_SECRET_ID=your_secret_id_here
COS_SECRET_KEY=your_secret_key_here
COS_BUCKET=your_bucket_name
COS_REGION=ap-shanghai
COS_PATH=/upload/neo4j

# ==================== 日志配置 ====================
LOG_DIR=/path/to/logs
```

## 配置项说明

### LLM配置

| 环境变量 | 说明 | 默认值 | 必填 |
|---------|------|--------|------|
| `SILICONFLOW_API_KEY` | SiliconFlow API密钥 | - | ✅ |
| `LLM_MODEL` | 大语言模型名称 | Qwen/Qwen2.5-7B-Instruct | ❌ |
| `EMBEDDING_MODEL` | 嵌入模型名称 | BAAI/bge-m3 | ❌ |
| `LLM_TIMEOUT` | 请求超时时间（秒） | 120 | ❌ |
| `MAX_TOKENS` | 最大输出token数 | 500 | ❌ |
| `MAX_RETRIES` | 最大重试次数 | 3 | ❌ |
| `LLM_TEMPERATURE` | 温度参数 | 0.0 | ❌ |

### Neo4j配置

| 环境变量 | 说明 | 默认值 | 必填 |
|---------|------|--------|------|
| `NEO4J_USERNAME` | Neo4j用户名 | neo4j | ❌ |
| `NEO4J_PASSWORD` | Neo4j密码 | 12345678 | ⚠️ 建议修改 |
| `NEO4J_URL` | Neo4j连接URL | bolt://localhost:7687 | ❌ |
| `NEO4J_DATABASE` | Neo4j数据库名 | neo4j | ❌ |

### 文档处理配置

| 环境变量 | 说明 | 默认值 | 必填 |
|---------|------|--------|------|
| `DOCUMENT_PATH` | 文档目录路径 | /Users/whalefall/Downloads/neo4j-data | ❌ |
| `MAX_PATHS_PER_CHUNK` | 每块最大路径数 | 3 | ❌ |
| `NUM_WORKERS` | 工作线程数 | 1 | ❌ |

### 重排序配置（可选功能）

| 环境变量 | 说明 | 默认值 | 必填 |
|---------|------|--------|------|
| `RERANK_API_KEY` | 重排序API密钥 | - | ❌ |

> **注意**: 重排序功能默认禁用，如需启用，需要在 `config.py` 中设置：
> ```python
> RERANK_CONFIG['enable'] = True
> ```

### 腾讯云COS配置

| 环境变量 | 说明 | 默认值 | 必填 |
|---------|------|--------|------|
| `COS_SECRET_ID` | 腾讯云SecretId | - | ✅ |
| `COS_SECRET_KEY` | 腾讯云SecretKey | - | ✅ |
| `COS_BUCKET` | COS存储桶名称 | - | ✅ |
| `COS_REGION` | COS地域 | ap-shanghai | ❌ |
| `COS_PATH` | COS存储路径 | /upload/neo4j | ❌ |

### 日志配置

| 环境变量 | 说明 | 默认值 | 必填 |
|---------|------|--------|------|
| `LOG_DIR` | 日志目录路径 | /Users/whalefall/Documents/workspace/python_demo/logs | ❌ |

## 安全建议

1. **不要提交 .env 文件到版本控制系统**
   ```bash
   # 在 .gitignore 中添加
   .env
   .env.local
   ```

2. **使用强密码**
   - Neo4j密码建议使用复杂密码
   - 定期更换API密钥

3. **限制文件权限**
   ```bash
   chmod 600 .env
   ```

4. **生产环境使用环境变量**
   - 不要在生产环境使用 .env 文件
   - 使用系统环境变量或密钥管理服务

## 验证配置

运行以下命令验证配置是否正确：

```bash
python3 -c "
from config import API_CONFIG, NEO4J_CONFIG, DOCUMENT_CONFIG
print('✅ 配置加载成功')
print(f'LLM模型: {API_CONFIG[\"siliconflow\"][\"llm_model\"]}')
print(f'Neo4j URL: {NEO4J_CONFIG[\"url\"]}')
print(f'文档路径: {DOCUMENT_CONFIG[\"path\"]}')
print(f'支持的文件类型: {DOCUMENT_CONFIG[\"supported_extensions\"]}')
"
```

## 故障排除

### 问题1: 环境变量未生效
**原因**: 可能是 `.env` 文件路径不正确或未加载

**解决方案**:
```python
# 确认 .env 文件位置
import os
from pathlib import Path
print(Path.cwd() / '.env')

# 手动指定 .env 文件路径
from dotenv import load_dotenv
load_dotenv('/path/to/.env')
```

### 问题2: 配置值为空
**原因**: 环境变量名称拼写错误或未设置

**解决方案**:
```python
import os
print(os.getenv('SILICONFLOW_API_KEY'))  # 检查是否能读取
```

### 问题3: 权限错误
**原因**: .env 文件权限过于开放

**解决方案**:
```bash
chmod 600 .env
```

## 相关文档
- [README.md](./README.md) - 项目使用说明
- [CHANGELOG.md](./CHANGELOG.md) - 更新日志
- [OPTIMIZATION_SUMMARY.md](./OPTIMIZATION_SUMMARY.md) - 代码优化总结

