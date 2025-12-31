# 知识图谱API服务

提供文件上传和知识图谱构建的RESTful接口服务。

## 安装依赖

```bash
pip install flask flask-cors werkzeug
```

## 启动服务

```bash
python api_server.py
```

服务将在 http://localhost:5000 启动

## API接口文档

### 1. 健康检查

**GET** `/health`

检查服务状态

**响应示例**:
```json
{
    "status": "healthy",
    "timestamp": "2025-12-24T10:30:00",
    "builder_initialized": true
}
```

### 2. 文件上传

**POST** `/upload`

上传文档文件用于知识图谱构建

**请求格式**: `multipart/form-data`

**参数**:
- `file`: 文档文件 (支持: txt, docx, pdf, md, json)

**响应示例**:
```json
{
    "message": "文件上传成功",
    "filename": "document.txt",
    "file_path": "/tmp/tmp123/document.txt",
    "temp_dir": "/tmp/tmp123"
}
```

### 3. 构建知识图谱

**POST** `/build_knowledge_graph`

根据上传的文件构建知识图谱

**请求格式**: `application/json`

**参数**:
```json
{
    "file_path": "/tmp/tmp123/document.txt"
}
```

**响应示例**:
```json
{
    "message": "知识图谱构建成功",
    "graph_id": "graph_1735123456",
    "document_count": 1,
    "processing_time": 45.23
}
```

### 4. 查询知识图谱

**POST** `/query`

查询已构建的知识图谱

**请求格式**: `application/json`

**参数**:
```json
{
    "query": "京东的创始人是谁？"
}
```

**响应示例**:
```json
{
    "query": "京东的创始人是谁？",
    "result": "京东的创始人是刘强东。",
    "timestamp": "2025-12-24T10:35:00"
}
```

### 5. 清理临时文件

**POST** `/clear_temp`

清理临时文件

**请求格式**: `application/json`

**参数**:
```json
{
    "temp_dir": "/tmp/tmp123"
}
```

## 使用示例

### 完整流程示例

```python
import requests
import json

# 1. 上传文件
with open('document.txt', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:5000/upload', files=files)
    file_data = response.json()

# 2. 构建知识图谱
payload = {'file_path': file_data['file_path']}
response = requests.post('http://localhost:5000/build_knowledge_graph', json=payload)
graph_data = response.json()

# 3. 查询知识图谱
query_payload = {'query': '公司的创始人是谁？'}
response = requests.post('http://localhost:5000/query', json=query_payload)
result = response.json()
```

### 使用客户端脚本

```bash
# 测试API服务
python api_client.py

# 使用指定文件测试
python api_client.py /path/to/your/document.txt
```

## 错误处理

所有接口在出错时都会返回相应的HTTP状态码和错误信息:

```json
{
    "error": "错误描述信息"
}
```

常见错误:
- `400`: 请求参数错误
- `404`: 接口不存在
- `500`: 服务器内部错误

## 注意事项

1. 文件大小限制: 16MB
2. 支持文件类型: txt, docx, pdf, md, json
3. 知识图谱构建可能需要几分钟时间
4. 临时文件会自动清理，也可以手动调用清理接口