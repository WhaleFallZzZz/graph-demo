# PDF 支持更新说明

## 更新日期
2025-12-30

## 更新内容

### ✅ 已添加 PDF 文件支持

系统现在支持处理 PDF 文件，与 txt 和 docx 文件一样可以用于构建知识图谱。

## 修改的文件

### 1. `config.py`
- **第64行**: 在 `supported_extensions` 中添加了 `".pdf"`
- 现在支持的文件类型：`[".txt", ".docx", ".pdf"]`

### 2. `README.md`
- 更新了功能特性说明，文档支持从 "txt, docx" 改为 "txt, docx, pdf"
- 在环境要求中添加了 `pypdf` 库的安装说明
- 添加了关于 PDF 解析库的使用说明

### 3. `requirements.txt` (新建)
- 创建了完整的依赖列表文件
- 包含了 `pypdf>=3.0.0` 用于 PDF 文件解析
- 包含了所有项目依赖

### 4. `file_type_detector.py`
- 已经支持 PDF 检测（无需修改）
- 第26行已包含 'pdf' 在允许的扩展名列表中
- 第40行已包含 PDF 的 MIME 类型映射

## 安装说明

### 方式一：安装单个库
如果您已经安装了其他依赖，只需安装 PDF 解析库：

```bash
pip install pypdf
```

### 方式二：使用 requirements.txt（推荐）
安装所有项目依赖：

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 准备 PDF 文件
将 PDF 文件放入配置的文档目录中：

```bash
# 默认目录
/Users/whalefall/Downloads/neo4j-data/

# 或使用环境变量配置的目录
export DOCUMENT_PATH=/your/custom/path
```

### 2. 运行程序
PDF 文件会自动被识别和处理，无需额外配置：

```bash
python main.py
```

或通过 API 上传：

```bash
curl -X POST http://localhost:8001/upload_and_build_sse \
  -F "file=@your_document.pdf"
```

## 技术细节

### PDF 解析
- 使用 `pypdf` 库进行 PDF 文本提取
- LlamaIndex 的 `SimpleDirectoryReader` 会自动处理 PDF 文件
- 支持文本型 PDF（扫描版 PDF 需要 OCR 处理，暂不支持）

### 文件类型检测
- 通过文件扩展名识别（`.pdf`）
- 通过 MIME 类型识别（`application/pdf`）
- 自动验证文件类型是否在允许列表中

### 支持的 PDF 特性
- ✅ 文本提取
- ✅ 多页文档
- ✅ 基本格式保留
- ❌ 图片内容（需要 OCR）
- ❌ 表格结构（会转为文本）
- ❌ 加密 PDF（需要密码解密）

## 注意事项

1. **文本型 PDF**: 只支持包含可提取文本的 PDF，扫描版 PDF 需要先进行 OCR 处理

2. **文件大小**: 建议单个 PDF 文件不超过 10MB，过大的文件可能导致处理缓慢

3. **编码问题**: 某些 PDF 可能存在编码问题，导致提取的文本乱码

4. **复杂格式**: 包含复杂表格、图表的 PDF 可能无法完美提取结构信息

## 测试建议

### 测试步骤
1. 准备一个简单的文本型 PDF 文件
2. 将文件放入文档目录
3. 运行程序并观察日志
4. 检查 Neo4j 中是否正确创建了知识图谱

### 测试示例
```python
# 测试文件类型检测
from file_type_detector import detect_file_type

result = detect_file_type('/path/to/test.pdf')
print(result)
# 预期输出: {'type': 'pdf', 'allowed': True, ...}
```

## 故障排除

### 问题1: 无法识别 PDF 文件
**解决方案**:
- 检查文件扩展名是否为 `.pdf`
- 确认文件不是损坏的
- 查看日志中的文件类型检测结果

### 问题2: PDF 提取文本为空
**解决方案**:
- 确认 PDF 是文本型而非扫描版
- 尝试用其他工具打开 PDF 并复制文本
- 考虑使用 OCR 工具预处理扫描版 PDF

### 问题3: 中文 PDF 乱码
**解决方案**:
- 检查 PDF 的字体嵌入情况
- 尝试使用 PDF 编辑工具重新保存
- 确认 `pypdf` 版本是最新的

## 后续优化建议

1. **OCR 支持**: 集成 OCR 引擎处理扫描版 PDF
2. **表格提取**: 使用专门的库提取表格结构
3. **图片处理**: 提取并分析 PDF 中的图片内容
4. **加密支持**: 添加密码解密功能
5. **分块优化**: 针对大型 PDF 优化分块策略

## 相关文档
- [LlamaIndex 文档加载器](https://docs.llamaindex.ai/en/stable/module_guides/loading/simpledirectoryreader.html)
- [pypdf 文档](https://pypdf.readthedocs.io/)
- [OPTIMIZATION_SUMMARY.md](./OPTIMIZATION_SUMMARY.md) - 代码优化总结

