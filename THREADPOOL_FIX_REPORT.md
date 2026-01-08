# DynamicThreadPool 错误修复报告

## 问题描述

项目启动时出现以下错误：

```
Exception in thread SchedulerThread:
Exception in thread MonitorThread:
2026-01-08 12:38:04,984 - llama.common.concurrent_utils - INFO - 动态线程池初始化: min=2, max=1, current=2
Traceback (most recent call last):
  File "/opt/homebrew/Cellar/python@3.14/3.14.2/Frameworks/Python.framework/Versions/3.14/lib/python3.14/threading.py", line 1082, in _bootstrap_inner
    self._context.run(self.run)
    ~~~~~~~~~~~~~~~~~^^^^^^^^^^
  File "/opt/homebrew/Cellar/python@3.14/3.14.2/Frameworks/Python.framework/Versions/3.14/lib/python3.14/threading.py", line 1024, in run
    self._target(*self._args, **self._kwargs)
    ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/whalefall/Documents/workspace/python_demo/llama/common/concurrent_utils.py", line 125, in _monitor_loop
    while not self._shutdown:
              ^^^^^^^^^^^^^^
AttributeError: 'DynamicThreadPool' object has no attribute '_shutdown'. Did you mean: 'shutdown'?
```

## 根本原因分析

### 问题 1: `_shutdown` 属性初始化时机错误

在 `DynamicThreadPool.__init__()` 方法中，调度线程和监控线程在 `_shutdown` 属性初始化之前就启动了：

```python
# 错误的代码顺序
self.scheduler_thread = threading.Thread(
    target=self._scheduler_loop,
    daemon=True,
    name="SchedulerThread"
)
self.scheduler_thread.start()

self.monitor_thread = threading.Thread(
    target=self._monitor_loop,
    daemon=True,
    name="MonitorThread"
)
self.monitor_thread.start()

# 关闭标志（在线程启动后才初始化）
self._shutdown = False
```

当线程启动并执行 `_scheduler_loop()` 和 `_monitor_loop()` 时，它们立即访问 `self._shutdown`，但此时该属性还不存在，导致 `AttributeError`。

### 问题 2: 参数配置不合理

日志显示 `min=2, max=1`，说明 `max_workers` 小于 `min_workers`，这是不合理的配置。这可能是由于：

1. `config.py` 中 `DOCUMENT_CONFIG["num_workers"]` 的默认值为 1
2. `kg_manager.py` 中使用 `min_workers=2`

导致 `max_workers=1 < min_workers=2`。

## 修复方案

### 修复 1: 调整 `_shutdown` 属性初始化顺序

将 `_shutdown` 属性的初始化移到线程启动之前：

```python
# 正确的代码顺序
# 关闭标志（必须在启动线程之前初始化）
self._shutdown = False

# 调度线程
self.scheduler_thread = threading.Thread(
    target=self._scheduler_loop,
    daemon=True,
    name="SchedulerThread"
)
self.scheduler_thread.start()

# 监控线程
self.monitor_thread = threading.Thread(
    target=self._monitor_loop,
    daemon=True,
    name="MonitorThread"
)
self.monitor_thread.start()
```

### 修复 2: 添加参数验证

在 `DynamicThreadPool.__init__()` 中添加参数验证逻辑：

```python
# 确定最大线程数
if max_workers is None:
    max_workers = max(4, (os.cpu_count() or 1) * 2)

# 参数验证：确保 max_workers >= min_workers
if max_workers < min_workers:
    logger.warning(f"max_workers ({max_workers}) 小于 min_workers ({min_workers})，自动调整为 {min_workers}")
    max_workers = min_workers
```

### 修复 3: 调整默认配置

修改 `config.py` 中 `DOCUMENT_CONFIG["num_workers"]` 的默认值：

```python
# 修改前
"num_workers": int(os.getenv("DOCUMENT_NUM_WORKERS", "1")),

# 修改后
"num_workers": int(os.getenv("DOCUMENT_NUM_WORKERS", "4")),
```

## 修复的文件

### 1. `/Users/whalefall/Documents/workspace/python_demo/llama/common/concurrent_utils.py`

**修改位置**: `DynamicThreadPool.__init__()` 方法

**修改内容**:
- 将 `_shutdown` 属性初始化移到线程启动之前
- 添加参数验证逻辑，确保 `max_workers >= min_workers`

### 2. `/Users/whalefall/Documents/workspace/python_demo/llama/config.py`

**修改位置**: `DOCUMENT_CONFIG` 字典

**修改内容**:
- 将 `num_workers` 的默认值从 1 改为 4

## 测试验证

### 测试文件

创建了以下测试文件来验证修复：

1. `test_threadpool_fix.py` - 完整的功能测试
2. `test_threadpool_fix_simple.py` - 简化的修复验证测试

### 测试结果

所有测试均通过：

```
✓ _shutdown 属性初始化修复: 通过
✓ 参数验证修复: 通过
✓ 原始错误场景 (min=2, max=1): 通过
```

### 测试覆盖的场景

1. **基本功能测试**: 验证线程池的基本功能正常
2. **参数验证测试**: 验证当 `max_workers < min_workers` 时自动调整
3. **_shutdown 属性测试**: 验证 `_shutdown` 属性正确初始化和使用
4. **原始错误场景测试**: 模拟原始错误配置 `min=2, max=1`

## 预期效果

修复后，项目启动时：

1. **不再出现 `AttributeError`**: `_shutdown` 属性在线程启动前已初始化
2. **参数自动调整**: 当 `max_workers < min_workers` 时，自动将 `max_workers` 调整为 `min_workers`
3. **日志输出清晰**: 会显示参数调整的警告信息
4. **线程池正常工作**: 调度线程和监控线程能够正常运行

## 日志示例

修复后的日志输出：

```
2026-01-08 12:44:28,637 - llama.common.concurrent_utils - WARNING - max_workers (1) 小于 min_workers (2)，自动调整为 2
2026-01-08 12:44:28,639 - llama.common.concurrent_utils - INFO - 动态线程池初始化: min=2, max=2, current=2
```

## 总结

通过调整属性初始化顺序和添加参数验证，成功解决了 `DynamicThreadPool` 的启动错误。修复后的代码更加健壮，能够处理不合理的参数配置，并提供了清晰的日志输出。所有测试均通过，项目应该能够正常启动。
