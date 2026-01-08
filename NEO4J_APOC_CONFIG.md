# Neo4j APOC 配置指南

## 问题描述
当前 Neo4j 数据库的 APOC 库处于沙箱模式，限制了某些过程的使用，导致 `apoc.refactor.mergeNodes` 等功能无法正常工作。

## 解决方案

### 方法 1: 修改 Neo4j 配置文件（推荐）

如果您有 Neo4j 数据库管理员权限，请按照以下步骤操作：

1. **找到 Neo4j 配置文件**
   - Linux/Mac: 通常在 `/etc/neo4j/neo4j.conf` 或 `conf/neo4j.conf`
   - Windows: 通常在 `conf/neo4j.conf`

2. **编辑配置文件**
   在 `neo4j.conf` 文件中添加以下配置：
   ```conf
   # 允许 APOC 所有过程不受限制
   dbms.security.procedures.unrestricted=apoc.*
   
   # 或者只允许特定的 APOC 过程（更安全）
   dbms.security.procedures.unrestricted=apoc.refactor.*
   ```

3. **重启 Neo4j 数据库**
   ```bash
   # Linux/Mac
   sudo systemctl restart neo4j
   
   # 或者使用 neo4j 命令
   neo4j restart
   ```

4. **验证配置**
   运行以下命令验证 APOC 是否可用：
   ```bash
   python3 diagnose_apoc.py
   ```

### 方法 2: 使用 Docker 运行 Neo4j

如果您使用 Docker 运行 Neo4j，可以在启动时添加环境变量：

```bash
docker run \
    --name neo4j \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/your_password \
    -e NEO4J_dbms_security_procedures_unrestricted=apoc.* \
    -e NEO4J_dbms_security_procedures_allowlist=apoc.* \
    neo4j:latest
```

### 方法 3: 使用 Neo4j Desktop

如果您使用 Neo4j Desktop：

1. 打开 Neo4j Desktop
2. 选择您的数据库
3. 点击 "Settings" 标签
4. 添加以下配置：
   ```
   dbms.security.procedures.unrestricted=apoc.*
   ```
5. 点击 "Apply" 并重启数据库

## 验证 APOC 配置

运行诊断工具验证 APOC 是否正常工作：

```bash
python3 diagnose_apoc.py
```

预期输出应该显示：
```
✓ APOC 已安装，版本: <version>
✓ apoc.meta.data 可用
✓ apoc.refactor.mergeNodes 可用
```

## 当前代码的回退机制

如果无法修改 Neo4j 配置，代码中已经实现了回退机制：

1. 当 `apoc.refactor.mergeNodes` 失败时，会自动使用手动 Cypher 查询来合并节点
2. 手动合并逻辑会：
   - 转移所有入边和出边
   - 保留边的属性
   - 删除被合并的节点

这个回退机制可以确保即使 APOC 不可用，代码也能正常工作。

## 注意事项

1. **安全性**: 允许 APOC 不受限制可能会带来安全风险，请确保在受信任的环境中运行
2. **权限**: 修改 Neo4j 配置需要管理员权限
3. **测试**: 在生产环境应用配置更改前，请先在测试环境验证

## 相关文件

- 诊断工具: `/Users/whalefall/Documents/workspace/python_demo/diagnose_apoc.py`
- 合并逻辑: `/Users/whalefall/Documents/workspace/python_demo/llama/kg_manager.py` (第 795-834 行)