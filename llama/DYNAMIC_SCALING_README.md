# 动态资源分配系统使用说明

## 概述

动态资源分配系统是一个智能的资源管理组件，能够根据活跃Worker数量自动调整资源分配。当只有一个Worker活跃时，系统会自动将其他空闲Worker的资源分配给活跃Worker，从而实现资源的最优利用和性能提升。

## 核心功能

### 1. Worker活动监控
- 实时监控所有Worker的活动状态
- 支持活动超时检测（默认30秒）
- 支持多种存储后端（文件/Redis）

### 2. 动态资源分配
- 根据活跃Worker数量自动调整资源
- 单Worker活跃时分配全部资源
- 多Worker活跃时平均分配资源
- 支持并发请求数、RPM限制、TPM限制等参数调整

### 3. 资源分配回调
- 支持自定义回调函数
- 资源分配变更时自动触发
- 可用于动态更新配置参数

## 集成方式

### 1. 在server.py中集成

动态资源分配系统已经集成到`server.py`中，启动服务器时会自动初始化：

```python
# 初始化动态资源分配系统
initialize_dynamic_scaling()
```

### 2. 在任务处理时更新Worker状态

在`build_graph_with_progress`函数中，系统会自动更新Worker活动状态：

```python
# 任务开始时标记为活跃
if scaling_manager:
    scaling_manager.update_activity(is_active=True, current_load=1.0, active_tasks=1)

# 任务结束时标记为非活跃
if scaling_manager:
    scaling_manager.update_activity(is_active=False, current_load=0.0, active_tasks=0)
```

### 3. 查看资源分配状态

通过API接口查看当前资源分配状态：

```bash
GET /scaling_status
```

返回示例：

```json
{
  "worker_id": "worker_12345",
  "total_workers": 4,
  "active_workers": 1,
  "active_worker_ids": ["worker_12345"],
  "is_scaling_enabled": true,
  "base_allocation": {
    "max_concurrent_requests": 10,
    "rpm_limit": 200,
    "tpm_limit": 10000,
    "num_workers": 10
  },
  "current_allocation": {
    "max_concurrent_requests": 40,
    "rpm_limit": 800,
    "tpm_limit": 40000,
    "num_workers": 40
  },
  "utilization_ratio": {
    "concurrent": 4.0,
    "rpm": 4.0,
    "tpm": 4.0,
    "workers": 4.0
  }
}
```

## 配置参数

### 基础配置

在`config.py`中的`RATE_LIMIT_CONFIG`定义了基础资源分配：

```python
RATE_LIMIT_CONFIG = {
    "request_delay": 0.1,
    "max_concurrent_requests": 10,  # 每个worker的并发数 (40/4=10)
    "retry_delay": 3.0,
    "rpm_limit": 200,  # 每个worker的RPM限制 (800/4=200)
    "tpm_limit": 10000,  # 每个worker的TPM限制 (40000/4=10000)
    "max_tokens_per_request": 4096,
    "max_retries": 3
}
```

### 动态调整参数

在`server.py`中的`initialize_dynamic_scaling`函数中可以调整：

```python
# 获取总Worker数量
total_workers = int(os.getenv('WORKER_COUNT', '4'))

# 创建基础资源分配配置
base_allocation = ResourceAllocation(
    max_concurrent_requests=RATE_LIMIT_CONFIG['max_concurrent_requests'],
    rpm_limit=RATE_LIMIT_CONFIG['rpm_limit'],
    tpm_limit=RATE_LIMIT_CONFIG['tpm_limit'],
    num_workers=3  # executor的max_workers
)
```

## 工作原理

### 资源分配算法

1. **单Worker活跃**：
   - 活跃Worker数 = 1
   - 分配全部资源：max_concurrent_requests = 40, rpm_limit = 800, tpm_limit = 40000

2. **多Worker活跃**：
   - 活跃Worker数 = N (N > 1)
   - 平均分配资源：每个Worker获得 total_resources / N

3. **无Worker活跃**：
   - 活跃Worker数 = 0
   - 使用基础分配：max_concurrent_requests = 10, rpm_limit = 200, tpm_limit = 10000

### 监控机制

- **监控间隔**：每5秒检查一次Worker状态
- **调整间隔**：每10秒调整一次资源分配
- **活动超时**：30秒未活动则视为空闲

### 状态同步

- **文件存储**：使用`.worker_activity_state.json`文件存储Worker状态
- **Redis存储**：支持使用Redis作为共享存储（需要配置环境变量）
- **原子写入**：使用临时文件+原子替换确保数据一致性

## 使用场景

### 场景1：低负载时段

- 只有一个Worker处理请求
- 系统自动分配全部资源给该Worker
- 处理速度提升4倍

### 场景2：高负载时段

- 多个Worker同时处理请求
- 系统自动平均分配资源
- 确保所有Worker公平获取资源

### 场景3：负载波动

- Worker数量动态变化
- 系统实时调整资源分配
- 自动适应负载变化

## 测试

运行测试脚本验证功能：

```bash
python3 test_dynamic_scaling.py
```

测试场景包括：
1. 单Worker场景
2. 多Worker场景
3. Worker超时场景
4. 资源分配回调

## 注意事项

1. **文件权限**：确保`.worker_activity_state.json`文件有读写权限
2. **环境变量**：可以通过环境变量配置Worker ID和数量
3. **日志监控**：关注日志中的资源分配调整信息
4. **性能影响**：监控间隔和调整间隔会影响响应速度

## 故障排查

### 问题1：资源分配未生效

**原因**：Worker状态未正确同步

**解决**：
- 检查`.worker_activity_state.json`文件是否存在
- 查看日志中的Worker状态信息
- 确认Worker ID配置正确

### 问题2：文件写入失败

**原因**：文件权限或并发写入冲突

**解决**：
- 检查文件权限
- 确保只有一个进程在写入
- 考虑使用Redis作为存储后端

### 问题3：回调函数未调用

**原因**：回调函数未正确设置

**解决**：
- 确认`set_allocation_callback`已调用
- 检查回调函数是否有异常
- 查看日志中的错误信息

## 性能优化建议

1. **调整监控间隔**：根据实际负载调整监控和调整间隔
2. **使用Redis**：在高并发场景下使用Redis提高性能
3. **缓存状态**：避免频繁读取文件
4. **异步处理**：将状态更新改为异步操作

## 总结

动态资源分配系统能够智能地根据活跃Worker数量调整资源分配，在低负载时提升性能，在高负载时保证公平。系统已经完全集成到你的知识图谱构建服务中，无需额外配置即可使用。
