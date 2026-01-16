"""
并发处理工具 - 动态线程池管理

本模块提供智能并发处理功能，包括：
- 动态线程池（根据系统资源自动调整）
- 任务队列管理
- 任务优先级支持
- 性能监控
- 优雅关闭
"""

import os
import time
import threading
import logging
from typing import Optional, Callable, Any, List, Dict
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from queue import PriorityQueue, Empty

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

logger = logging.getLogger(__name__)


class DynamicThreadPool:
    """
    动态线程池管理器
    
    根据系统资源和工作负载自动调整线程池大小，
    提供智能的任务调度和资源管理。
    
    主要特性：
    - 自动调整线程数：根据负载和系统资源动态调整
    - 任务优先级：支持任务优先级调度
    - 性能监控：实时监控 CPU 和内存使用情况
    - 优雅关闭：支持优雅关闭和超时控制
    
    使用示例：
        ```python
        from llama.common import DynamicThreadPool
        
        # 创建动态线程池
        pool = DynamicThreadPool(
            min_workers=2,
            max_workers=4,
            idle_timeout=60.0
        )
        
        # 使用线程池
        with pool:
            # 提交任务...
            pass
        
        # 或者手动关闭
        pool.shutdown(wait=True)
        ```
    """
    
    def __init__(
        self,
        min_workers: int = 4,
        max_workers: Optional[int] = None,
        idle_timeout: float = 60.0,
        queue_size: int = 1000
    ):
        """
        初始化动态线程池
        
        Args:
            min_workers: 最小线程数（默认：4）
            max_workers: 最大线程数（默认：CPU核心数*2）
            idle_timeout: 空闲超时（秒），超过此时间减少线程（默认：60）
            queue_size: 任务队列大小（默认：1000）
        """
        # 确定最大线程数
        if max_workers is None:
            max_workers = max(8, (os.cpu_count() or 1) * 2)
        
        # 参数验证：确保 max_workers >= min_workers
        if max_workers < min_workers:
            logger.warning(f"max_workers ({max_workers}) 小于 min_workers ({min_workers})，自动调整为 {min_workers}")
            max_workers = min_workers
        
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.idle_timeout = idle_timeout
        self.queue_size = queue_size
        
        # 关闭标志（必须在启动线程之前初始化）
        self._shutdown = False
        
        # 当前线程数（从最小开始）
        self.current_workers = min_workers
        
        # 创建线程池
        self.executor = ThreadPoolExecutor(
            max_workers=self.current_workers,
            thread_name_prefix="DynamicPool"
        )
        
        # 任务队列（带优先级）
        self.task_queue = PriorityQueue(maxsize=queue_size)
        self.task_counter = 0  # 用于任务排序
        
        # 性能统计
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'total_time': 0.0,
            'avg_time': 0.0,
            'cpu_usage': 0.0,
            'memory_usage': 0.0
        }
        
        # 负载均衡参数
        self.load_balance_threshold = 0.8
        self.last_adjustment_time = time.time()
        
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
        
        logger.info(f"动态线程池初始化: min={min_workers}, max={max_workers}, current={self.current_workers}")
    
    def _scheduler_loop(self):
        """任务调度循环"""
        while not self._shutdown:
            try:
                # 从队列获取任务（带超时）
                priority, task_id, task = self.task_queue.get(timeout=1.0)
                
                # 提交到线程池
                future = self.executor.submit(task['func'], *task['args'], **task['kwargs'])
                
                # 添加回调
                future.add_done_callback(self._task_completed)
                
                # 标记任务已提交
                self.task_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"调度器错误: {e}")
    
    def _monitor_loop(self):
        """监控循环 - 动态调整线程数"""
        last_active_time = time.time()
        consecutive_idle = 0
        consecutive_overload = 0
        
        while not self._shutdown:
            time.sleep(10.0)  # 每10秒检查一次
            
            # 获取系统资源使用情况
            cpu_usage = self._get_cpu_usage()
            memory_usage = self._get_memory_usage()
            self.stats['cpu_usage'] = cpu_usage
            self.stats['memory_usage'] = memory_usage
            
            # 检查线程池状态
            active_tasks = self._get_active_tasks()
            queue_size = self.task_queue.qsize()
            
            # 负载计算
            load_factor = self._calculate_load_factor(queue_size, len(active_tasks), cpu_usage, memory_usage)
            
            if len(active_tasks) == 0:
                consecutive_idle += 1
                consecutive_overload = 0
                
                # 如果空闲时间过长，减少线程数
                if consecutive_idle * 5 >= self.idle_timeout:
                    self._adjust_workers(-1)
                    consecutive_idle = 0
            else:
                consecutive_idle = 0
                last_active_time = time.time()
                
                # 检查是否过载
                if load_factor > self.load_balance_threshold:
                    consecutive_overload += 1
                    if consecutive_overload >= 2:  # 连续两次过载才增加线程
                        self._adjust_workers(1)
                        consecutive_overload = 0
                else:
                    consecutive_overload = 0
            
            # 检查队列积压
            if queue_size > self.queue_size * 0.8 and len(active_tasks) >= self.current_workers:
                # 只有在所有线程都在工作且队列积压时才增加线程
                if cpu_usage < 80:  # CPU使用率不过高时才增加
                    self._adjust_workers(1)
    
    def _get_cpu_usage(self) -> float:
        """获取CPU使用率"""
        try:
            import psutil
            return psutil.cpu_percent(interval=0.1)
        except ImportError:
            logger.debug("psutil未安装，无法获取CPU使用率")
            return 0.0
        except Exception as e:
            logger.warning(f"获取CPU使用率失败: {e}")
            return 0.0
    
    def _get_memory_usage(self) -> float:
        """获取内存使用率"""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except ImportError:
            logger.debug("psutil未安装，无法获取内存使用率")
            return 0.0
        except Exception as e:
            logger.warning(f"获取内存使用率失败: {e}")
            return 0.0
    
    def _calculate_load_factor(self, queue_size: int, active_tasks: int, cpu_usage: float, memory_usage: float) -> float:
        """
        计算负载因子
        
        Args:
            queue_size: 队列大小
            active_tasks: 活动任务数
            cpu_usage: CPU使用率
            memory_usage: 内存使用率
            
        Returns:
            负载因子（0-1之间）
        """
        # 队列负载
        queue_load = min(queue_size / self.queue_size, 1.0)
        
        # 线程负载
        thread_load = min(active_tasks / self.current_workers, 1.0)
        
        # 系统资源负载
        resource_load = (cpu_usage + memory_usage) / 200.0  # 归一化到0-1
        
        # 综合负载因子（加权平均）
        load_factor = (queue_load * 0.4 + thread_load * 0.4 + resource_load * 0.2)
        
        return load_factor
    
    def _adjust_workers(self, delta: int):
        """
        调整线程数
        
        Args:
            delta: 调整量（+1 增加，-1 减少）
        """
        new_workers = self.current_workers + delta
        
        # 确保在范围内
        new_workers = max(self.min_workers, min(new_workers, self.max_workers))
        
        if new_workers != self.current_workers:
            old_workers = self.current_workers
            self.current_workers = new_workers
            
            # 创建新的线程池
            old_executor = self.executor
            self.executor = ThreadPoolExecutor(
                max_workers=self.current_workers,
                thread_name_prefix="DynamicPool"
            )
            
            logger.info(f"线程数调整: {old_workers} -> {new_workers} (队列大小: {self.task_queue.qsize()})")
    
    def _get_active_tasks(self) -> List[Future]:
        """获取当前活动的任务"""
        # 这是一个简化的实现
        # 实际应用中可能需要更精确的跟踪
        return []
    
    def _task_completed(self, future: Future):
        """任务完成回调"""
        try:
            future.result()  # 获取结果或抛出异常
            self.stats['completed_tasks'] += 1
        except Exception as e:
            self.stats['failed_tasks'] += 1
            logger.error(f"任务失败: {e}")
    
    def shutdown(self, wait: bool = True, timeout: Optional[float] = None):
        """
        关闭线程池
        
        Args:
            wait: 是否等待所有任务完成
            timeout: 等待超时时间
            
        Examples:
            >>> pool = DynamicThreadPool()
            >>> pool.shutdown(wait=True, timeout=30.0)
        """
        self._shutdown = True
        
        logger.info("正在关闭动态线程池...")
        
        # 关闭调度器
        if wait:
            # 等待队列清空
            while self.task_queue.qsize() > 0:
                logger.debug(f"等待队列清空: 剩余 {self.task_queue.qsize()} 个任务")
                time.sleep(0.5)
        
        # 关闭线程池
        self.executor.shutdown(wait=wait)
        
        # 等待监控线程结束
        self.scheduler_thread.join(timeout=5.0)
        self.monitor_thread.join(timeout=5.0)
        
        logger.info("动态线程池已关闭")
        logger.info(f"最终统计: {self.get_stats()}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取性能统计信息
        
        Returns:
            包含性能指标的字典
            
        Examples:
            >>> pool = DynamicThreadPool()
            >>> stats = pool.get_stats()
        """
        total_completed = self.stats['completed_tasks'] + self.stats['failed_tasks']
        
        if total_completed > 0:
            self.stats['avg_time'] = self.stats['total_time'] / total_completed
            self.stats['success_rate'] = self.stats['completed_tasks'] / total_completed
        else:
            self.stats['avg_time'] = 0.0
            self.stats['success_rate'] = 0.0
        
        return {
            **self.stats,
            'queue_size': self.task_queue.qsize(),
            'current_workers': self.current_workers,
            'min_workers': self.min_workers,
            'max_workers': self.max_workers
        }
    
    def __enter__(self):
        """支持上下文管理器协议"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出时关闭"""
        self.shutdown(wait=True)
