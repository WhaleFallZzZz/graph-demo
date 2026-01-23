# build_graph_sse 接口工作流程说明

该接口用于根据文件的 COS URL 异步构建知识图谱，并通过 SSE (Server-Sent Events) 实时推送构建进度。

## 1. 架构概览

整个流程涉及 API 服务层、后台任务处理层和核心构建层：

- **API 层 (`server.py`)**: 接收请求，建立 SSE 连接，管理监听器。
- **服务层 (`graph_service.py`)**: 执行后台任务调度 (`ThreadPoolExecutor`)，管理文件下载与进度跟踪。
- **核心层 (`kg_manager.py`)**: 执行文档加载、语义分块、实体提取与图谱构建。

---

## 2. 详细流程步骤

### 阶段一：API 请求与初始化

1. **客户端发起请求**: 客户端调用 `POST /build_graph_sse`，提供 `file_url` (必选) 和 `file_name` (可选)。
2. **生成会话**: 服务器为请求生成唯一的 `client_id`。
3. **建立连接**: 返回 `text/event-stream` 响应，建立 SSE 长连接。
4. **启动后台任务**: 调用 `graph_service.submit_build_task`，将构建任务提交到线程池异步执行。

### 阶段二：后台处理 (GraphService)

进入 `build_graph_with_progress` 方法：

1. **进度跟踪**: 初始化 `ProgressTracker`，准备向 `progress_manager` 发送进度更新。
2. **文件下载**:
   - 从腾讯云 COS 下载文件到本地临时目录。
   - 解析/推断文件名及后缀（如从 URL 或 Content-Type 推断）。
3. **文档加载**: 调用 `builder.load_documents`。
   - 使用 `DeepSeekOCRParser` (针对 PDF) 或 `SimpleDirectoryReader` 加载内容。
   - **增量处理**: 检查文件哈希，避免重复处理未修改的文件。

### 阶段三：核心构建 (KnowledgeGraphManager)

1. **语义分块 (Chunking)**:
   - 使用 `ImprovedSemanticChunker` 进行分块。
   - 包含：结构化切分 (段落) -> 语义聚合 -> 重叠保留 (12% overlap)。
   - 针对医学术语进行边界保护，防止术语被由于切分而截断。
2. **图谱构建**: 调用 `builder.build_knowledge_graph`。
   - 遍历分块文档，调用 LLM (如 DeepSeek) 进行实体和关系提取。
   - 将结果存储到 Neo4j 图数据库中。
3. **建立索引**: 为加载的文档分块建立医学关键词倒排索引，加速后续检索。

### 阶段四：后处理与完成

1. **节点清洗**: 执行 `offline_node_cleaning.py`，规范化提取的实体。
2. **属性下沉**: 执行 `offline_property_sinking.py`，优化属性存储结构。
3. **任务结束**: 更新任务状态到全局 `task_results`，并将 `complete` 事件推送到 SSE 流。

---

## 3. SSE 进度推送机制

系统通过 `progress_manager` 实现解耦的进度推送：

- 核心步骤（加载、分块、提取等）调用 `progress_tracker.update_stage()`。
- `progress_manager` 将消息分发给对应的 `client_id` 监听器。
- `server.py` 中的 `consume_sse_queue` 实时从队列读取消息并 yield 格式化的 SSE 数据包。

**常见事件类型：**

- `progress`: 包含当前阶段、百分比进度和状态描述。
- `error`: 处理过程中发生的异常信息。
- `complete`: 构建成功后的汇总信息（节点数、耗时等）。
