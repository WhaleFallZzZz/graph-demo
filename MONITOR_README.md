# 知识图谱文件监控自动化脚本

## 功能概述

`knowledge_graph_monitor.py` 是一个自动化文件监控脚本，用于实时监控指定目录下的文件变化，自动触发知识图谱生成流程，并确保文件处理的幂等性（同一文件不重复生成）。

## 核心特性

1. **实时监控**：使用 `watchdog` 库监控目录下的文件创建和修改事件
2. **幂等性保证**：通过标记文件（`.processed`）确保同一文件不重复处理
3. **文件哈希标识**：使用 MD5 哈希值作为文件的唯一标识
4. **异常处理**：完善的错误处理和自动重试机制（最多3次）
5. **批量处理**：支持短时间内多个文件的批量处理
6. **网络检测**：在调用接口前检查网络连接状态
7. **日志记录**：详细的日志记录，按日期滚动生成

## 安装依赖

```bash
pip install watchdog requests
```

## 配置文件

脚本使用 `config.json` 作为配置文件，支持以下配置项：

```json
{
  "watch_directory": "./wait_build",           // 监控目录路径
  "upload_api": "http://localhost:5000/upload",  // 文件上传接口地址
  "build_graph_api": "http://localhost:5000/build_graph_sse",  // 知识图谱构建接口地址
  "supported_extensions": [".txt", ".md", ".docx"],  // 支持的文件扩展名
  "log_dir": "./logs",                        // 日志目录
  "max_retries": 3,                           // 最大重试次数
  "retry_delay": 5.0,                         // 重试延迟（秒）
  "batch_delay": 2.0,                         // 批量处理延迟（秒）
  "batch_size": 5                             // 批量处理大小
}
```

### 配置说明

- **watch_directory**: 要监控的目录路径，如果不存在会自动创建
- **upload_api**: 文件上传接口的完整URL
- **build_graph_api**: 知识图谱构建接口的完整URL（支持SSE流式响应）
- **supported_extensions**: 支持处理的文件扩展名列表（不区分大小写）
- **log_dir**: 日志文件存储目录
- **max_retries**: 接口调用失败时的最大重试次数
- **retry_delay**: 每次重试之间的延迟时间（秒）
- **batch_delay**: 批量处理延迟，用于等待短时间内多个文件变化（秒）
- **batch_size**: 每次批量处理的最大文件数量

## 使用方法

### 1. 基本使用

```bash
python knowledge_graph_monitor.py
```

### 2. 指定配置文件

```bash
python knowledge_graph_monitor.py --config /path/to/config.json
```

### 3. 重试失败的文件

```bash
python knowledge_graph_monitor.py --retry-failed
```

## 工作流程

1. **初始化**：检查监控目录是否存在，不存在则创建
2. **文件监控**：监控目录下的文件创建和修改事件
3. **文件过滤**：
   - 仅处理配置中指定的文件扩展名
   - 跳过隐藏文件和目录
   - 跳过已存在对应标记文件且状态为"已完成"的文件
4. **文件处理**：
   - 生成文件MD5哈希值
   - 创建处理标记文件（`.processed`）
   - 调用上传接口上传文件
   - 上传成功后，调用构建接口触发知识图谱生成
   - 处理成功后，更新标记文件状态为"已完成"
5. **异常处理**：
   - 接口调用失败时，记录错误信息
   - 自动重试（最多3次）
   - 重试失败后，标记文件状态为"失败"

## 标记文件格式

每个处理过的文件都会生成一个对应的标记文件（原文件名 + `.processed`），格式如下：

```json
{
  "file_hash": "abc123def456...",
  "status": "completed",
  "created_at": "2025-12-30T10:30:00",
  "updated_at": "2025-12-30T10:35:00"
}
```

### 状态说明

- **processing**: 正在处理中
- **completed**: 处理完成
- **failed**: 处理失败

## 日志文件

日志文件存储在配置的 `log_dir` 目录下，按日期命名：`monitor_YYYY-MM-DD.log`

日志包含以下信息：
- 时间戳
- 文件名称
- 处理状态
- 接口调用结果
- 错误信息

## 接口要求

### 上传接口 (`/upload`)

**请求方式**: POST  
**Content-Type**: multipart/form-data

**参数**:
- `file`: 文件对象（必需）
- `file_hash`: 文件哈希值（可选，建议传递）

**响应格式**:
```json
{
  "success": true,
  "message": "文件上传成功",
  "data": {
    "file_info": {
      "filename": "document.txt",
      "file_url": "https://...",
      "size": 1024,
      "md5": "...",
      "file_type": "text/plain"
    }
  }
}
```

### 构建接口 (`/build_graph_sse`)

**请求方式**: POST  
**Content-Type**: application/json

**参数**:
```json
{
  "file_url": "https://...",
  "file_hash": "abc123..."
}
```

**响应格式**: SSE (Server-Sent Events) 流式响应

## 注意事项

1. **幂等性保证**：
   - 已处理的文件（存在 `.processed` 标记且状态为 `completed`）不会重复处理
   - 文件修改后，如果内容哈希值变化，会删除旧标记并重新处理

2. **并发处理**：
   - 使用线程池处理文件，避免阻塞监控线程
   - 同一文件不会同时被多个线程处理

3. **网络检测**：
   - 在调用接口前会检查网络连接
   - 网络不可用时不会调用接口，避免不必要的错误

4. **批量处理**：
   - 短时间内多个文件变化时，会等待 `batch_delay` 秒后批量处理
   - 避免频繁的接口调用

5. **失败重试**：
   - 失败的文件不会自动重试（避免无限循环）
   - 使用 `--retry-failed` 参数手动触发重试
   - 或手动删除失败标记文件后，文件修改时会自动重新处理

## 故障排查

### 1. 文件未被处理

- 检查文件扩展名是否在 `supported_extensions` 中
- 检查是否已存在 `.processed` 标记文件
- 查看日志文件了解详细信息

### 2. 接口调用失败

- 检查接口地址是否正确
- 检查网络连接是否正常
- 查看日志文件中的错误信息
- 检查服务器是否正常运行

### 3. 标记文件损坏

- 如果标记文件损坏，可以手动删除，文件修改时会重新处理
- 或者使用 `--retry-failed` 参数重试失败的文件

## 性能优化建议

1. **批量处理**：调整 `batch_delay` 和 `batch_size` 参数，平衡响应速度和批量效率
2. **线程池大小**：默认使用3个工作线程，可根据实际情况调整
3. **日志级别**：生产环境可以调整日志级别为 WARNING，减少日志量

## 示例

### 示例1：监控本地目录

```bash
# 编辑 config.json
{
  "watch_directory": "/path/to/documents",
  "upload_api": "http://localhost:5000/upload",
  "build_graph_api": "http://localhost:5000/build_graph_sse"
}

# 启动监控
python knowledge_graph_monitor.py
```

### 示例2：重试失败的文件

```bash
# 重试所有失败的文件
python knowledge_graph_monitor.py --retry-failed
```

## 开发说明

脚本采用面向对象设计，主要类：

- **FileProcessor**: 文件处理器，负责文件上传和知识图谱构建
- **FileMonitorHandler**: 文件系统事件处理器，处理文件创建和修改事件
- **KnowledgeGraphMonitor**: 监控主类，协调各个组件

## 许可证

本脚本遵循项目整体许可证。