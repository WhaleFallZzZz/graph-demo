# 更新日志 (Changelog)

## [2025-12-30] - 修复 RERANK_CONFIG 导入错误

### 修复内容 🐛
- **添加 RERANK_CONFIG**: 在 `config.py` 中添加了缺失的 `RERANK_CONFIG` 配置
- **环境变量支持**: `RERANK_API_KEY` 现在支持从环境变量读取
- **文档完善**: 创建了 `ENV_CONFIG.md` 详细说明环境变量配置

### 修改文件 📝
1. **config.py**
   - 添加了 `RERANK_CONFIG` 配置项
   - 包含重排序功能的完整配置（默认禁用）

2. **ENV_CONFIG.md** (新建)
   - 完整的环境变量配置说明
   - 包含配置模板和安全建议

### 配置说明 📖
```python
RERANK_CONFIG = {
    "enable": False,  # 默认禁用
    "provider": "siliconflow",
    "api_key": os.getenv("RERANK_API_KEY", ""),
    "model": "BAAI/bge-reranker-v2-m3",
    "top_n": 5,
    "initial_top_k": 10,
}
```

### 验证结果 ✅
- ✅ `RERANK_CONFIG` 可以正常导入
- ✅ 所有配置项导入测试通过
- ✅ 代码 linter 检查通过

---

## [2025-12-30] - PDF 支持更新

### 新增功能 ✨
- **PDF 文件支持**: 系统现在支持处理 PDF 文件，可用于构建知识图谱
- **requirements.txt**: 创建了完整的项目依赖文件
- **测试脚本**: 添加了 `test_pdf_support.py` 用于验证 PDF 支持配置

### 修改文件 📝
1. **config.py**
   - 在 `DOCUMENT_CONFIG['supported_extensions']` 中添加了 `".pdf"`
   - 支持的文件类型：`[".txt", ".docx", ".pdf"]`

2. **README.md**
   - 更新功能特性说明，文档支持改为 "txt, docx, pdf"
   - 添加 `pypdf` 库安装说明
   - 更新环境要求部分

3. **requirements.txt** (新建)
   - 包含所有项目依赖
   - 添加 `pypdf>=3.0.0` 用于 PDF 解析

4. **file_type_detector.py**
   - 已包含 PDF 支持（无需修改）

### 新增文档 📚
- **PDF_SUPPORT_UPDATE.md**: PDF 支持详细说明文档
- **test_pdf_support.py**: PDF 支持测试脚本
- **CHANGELOG.md**: 项目更新日志（本文件）

### 使用说明 📖

#### 安装 PDF 支持
```bash
# 方式一：安装单个库
pip install pypdf

# 方式二：安装所有依赖（推荐）
pip install -r requirements.txt
```

#### 使用 PDF 文件
将 PDF 文件放入文档目录即可自动处理：
```bash
# 默认目录
/Users/whalefall/Downloads/neo4j-data/

# 或使用环境变量
export DOCUMENT_PATH=/your/custom/path
```

#### 验证配置
运行测试脚本：
```bash
python3 test_pdf_support.py
```

### 测试结果 ✅
- ✅ 配置检查通过
- ✅ 文件检测器通过
- ✅ 扩展名识别正常
- ✅ MIME 类型映射正常

### 注意事项 ⚠️
1. **仅支持文本型 PDF**: 扫描版 PDF 需要 OCR 处理
2. **文件大小限制**: 建议单个 PDF 不超过 10MB
3. **需要安装 pypdf**: 运行 `pip install pypdf`

---

## [2025-12-30] - 代码优化

### 优化内容 🔧
1. **清理死代码**: 删除了未使用的代码片段
2. **环境变量支持**: 所有敏感配置支持从环境变量读取
3. **错误处理**: 改进了异常处理和日志记录
4. **类型注解**: 为核心函数添加了完整的类型注解
5. **资源管理**: 优化了临时文件和资源清理
6. **代码重构**: 简化了复杂函数，提高可读性

### 详细说明
参见 [OPTIMIZATION_SUMMARY.md](./OPTIMIZATION_SUMMARY.md)

---

## 版本历史

### 当前版本
- **版本**: 1.1.0
- **日期**: 2025-12-30
- **主要特性**: 
  - 支持 txt, docx, pdf 文件
  - 环境变量配置
  - 完善的错误处理
  - 类型注解支持

### 未来计划 🚀
- [ ] OCR 支持处理扫描版 PDF
- [ ] 表格结构提取
- [ ] 图片内容分析
- [ ] 加密 PDF 支持
- [ ] 单元测试覆盖
- [ ] 性能优化和缓存机制

