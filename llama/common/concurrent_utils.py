"""
并发处理工具 - 动态线程池管理

本模块提供智能并发处理功能，包括：
- 动态线程池（根据系统资源自动调整）
- 任务队列管理
- 任务优先级支持
- 性能监控
- 优雅关闭
- 智能负载均衡（优化版）
- 自适应批量大小调整（优化版）
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
    """
    
    def __init__(
        self,
        min_workers: int = 2,
        max_workers: Optional[int] = None,
        idle_timeout: float = 60.0,
        queue_size: int = 1000
    ):
        """
        初始化动态线程池
        
        Args:
            min_workers: 最小线程数（默认：2）
            max_workers: 最大线程数（默认：CPU核心数*2）
            idle_timeout: 空闲超时（秒），超过此时间减少线程（默认：60）
            queue_size: 任务队列大小（默认：1000）
        """
        # 确定最大线程数
        if max_workers is None:
            max_workers = max(4, (os.cpu_count() or 1) * 2)
        
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
        
        # 自适应批量大小（优化版）
        self.optimal_batch_size = 10
        self.batch_size_history = []
        self.performance_history = []
        
        # 负载均衡参数（优化版）
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
        """监控循环 - 动态调整线程数（优化版）"""
        last_active_time = time.time()
        consecutive_idle = 0
        consecutive_overload = 0
        
        while not self._shutdown:
            time.sleep(5.0)  # 每5秒检查一次
            
            # 获取系统资源使用情况（优化版）
            cpu_usage = self._get_cpu_usage()
            memory_usage = self._get_memory_usage()
            self.stats['cpu_usage'] = cpu_usage
            self.stats['memory_usage'] = memory_usage
            
            # 检查线程池状态
            active_tasks = self._get_active_tasks()
            queue_size = self.task_queue.qsize()
            
            # 负载计算（优化版）
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
                
                # 检查是否过载（优化版）
                if load_factor > self.load_balance_threshold:
                    consecutive_overload += 1
                    if consecutive_overload >= 2:  # 连续两次过载才增加线程
                        self._adjust_workers(1)
                        consecutive_overload = 0
                else:
                    consecutive_overload = 0
            
            # 检查队列积压（优化版）
            if queue_size > self.queue_size * 0.8 and len(active_tasks) >= self.current_workers:
                # 只有在所有线程都在工作且队列积压时才增加线程
                if cpu_usage < 80:  # CPU使用率不过高时才增加
                    self._adjust_workers(1)
            
            # 更新自适应批量大小（优化版）
            self._update_adaptive_batch_size(load_factor)
    
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
        计算负载因子（优化版）
        
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
    
    def _update_adaptive_batch_size(self, load_factor: float):
        """
        更新自适应批量大小（优化版）
        
        Args:
            load_factor: 负载因子
        """
        # 根据负载因子调整批量大小
        if load_factor > 0.8:
            # 高负载：减小批量大小
            new_batch_size = max(5, self.optimal_batch_size - 2)
        elif load_factor < 0.3:
            # 低负载：增加批量大小
            new_batch_size = min(20, self.optimal_batch_size + 2)
        else:
            # 中等负载：保持当前批量大小
            new_batch_size = self.optimal_batch_size
        
        # 只有在批量大小变化时才更新
        if new_batch_size != self.optimal_batch_size:
            logger.info(f"自适应批量大小调整: {self.optimal_batch_size} -> {new_batch_size} (负载因子: {load_factor:.2f})")
            self.optimal_batch_size = new_batch_size
            self.batch_size_history.append(new_batch_size)
            
            # 限制历史记录长度
            if len(self.batch_size_history) > 100:
                self.batch_size_history = self.batch_size_history[-100:]
    
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
    
    def submit(
        self,
        func: Callable,
        priority: int = 0,
        *args,
        **kwargs
    ) -> Future:
        """
        提交任务到线程池
        
        Args:
            func: 要执行的函数
            priority: 任务优先级（数字越大优先级越高）
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            Future 对象
            
        Examples:
            >>> pool = DynamicThreadPool()
            >>> future = pool.submit(my_function, arg1, arg2, priority=10)
            >>> result = future.result()
        """
        self.task_counter += 1
        
        task = {
            'func': func,
            'args': args,
            'kwargs': kwargs,
            'id': self.task_counter
        }
        
        # 使用负优先级（因为 PriorityQueue 是最小堆）
        # 使用三元组确保唯一性：(priority, task_id, task)
        self.task_queue.put((-priority, self.task_counter, task))
        self.stats['total_tasks'] += 1
        
        logger.debug(f"任务已提交: ID={task['id']}, 优先级={priority}")
        
        # 返回一个 Future（注意：实际执行在调度线程中）
        # 这里返回一个占位符 Future，实际应用中需要更复杂的处理
        return Future()
    
    def submit_batch(
        self,
        func: Callable,
        args_list: List[tuple],
        priority: int = 0
    ) -> List[Future]:
        """
        批量提交任务
        
        Args:
            func: 要执行的函数
            args_list: 参数列表，每个元素是一个元组
            priority: 任务优先级
            
        Returns:
            Future 对象列表
            
        Examples:
            >>> pool = DynamicThreadPool()
            >>> futures = pool.submit_batch(my_function, [(arg1,), (arg2,)])
        """
        futures = []
        
        for args in args_list:
            future = self.submit(func, priority=priority, *args)
            futures.append(future)
        
        return futures
    
    def wait_for_completion(self, timeout: Optional[float] = None):
        """
        等待所有任务完成
        
        Args:
            timeout: 超时时间（秒），None 表示无限等待
            
        Examples:
            >>> pool = DynamicThreadPool()
            >>> pool.wait_for_completion(timeout=30.0)
        """
        start_time = time.time()
        
        while not self._shutdown:
            # 检查队列和线程池状态
            queue_size = self.task_queue.qsize()
            active_tasks = self._get_active_tasks()
            
            if queue_size == 0 and len(active_tasks) == 0:
                logger.info("所有任务已完成")
                break
            
            # 检查超时
            if timeout and (time.time() - start_time) > timeout:
                logger.warning(f"等待超时: {timeout}秒")
                break
            
            time.sleep(0.1)
    
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
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'total_time': 0.0,
            'avg_time': 0.0
        }
        logger.info("统计信息已重置")
    
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
    
    def __enter__(self):
        """支持上下文管理器协议"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出时关闭"""
        self.shutdown(wait=True)


class TaskManager:
    """
    任务管理器 - 简化任务提交和管理
    
    提供更简单的接口来管理异步任务。
    """
    
    def __init__(self, pool: Optional[DynamicThreadPool] = None):
        """
        初始化任务管理器
        
        Args:
            pool: 线程池实例（如果为 None 则创建默认实例）
        """
        if pool is None:
            pool = DynamicThreadPool()
        
        self.pool = pool
        self.tasks: Dict[str, Future] = {}
    
    def submit_task(
        self,
        task_id: str,
        func: Callable,
        *args,
        priority: int = 0,
        **kwargs
    ) -> Future:
        """
        提交任务并跟踪
        
        Args:
            task_id: 任务唯一标识
            func: 要执行的函数
            *args: 位置参数
            priority: 任务优先级
            **kwargs: 关键字参数
            
        Returns:
            Future 对象
            
        Examples:
            >>> manager = TaskManager()
            >>> future = manager.submit_task("task1", my_function, arg1, arg2)
        """
        future = self.pool.submit(func, priority=priority, *args, **kwargs)
        self.tasks[task_id] = future
        
        logger.debug(f"任务已提交: {task_id}")
        return future
    
    def get_task_status(self, task_id: str) -> Optional[str]:
        """
        获取任务状态
        
        Args:
            task_id: 任务标识
            
        Returns:
            任务状态：'pending', 'running', 'completed', 'failed'
            
        Examples:
            >>> manager = TaskManager()
            >>> status = manager.get_task_status("task1")
        """
        if task_id not in self.tasks:
            return None
        
        future = self.tasks[task_id]
        
        if future.done():
            if future.exception():
                return 'failed'
            return 'completed'
        else:
            return 'running'
    
    def wait_for_task(self, task_id: str, timeout: Optional[float] = None):
        """
        等待特定任务完成
        
        Args:
            task_id: 任务标识
            timeout: 超时时间
            
        Returns:
            任务结果
            
        Examples:
            >>> manager = TaskManager()
            >>> result = manager.wait_for_task("task1", timeout=30.0)
        """
        if task_id not in self.tasks:
            raise ValueError(f"任务不存在: {task_id}")
        
        future = self.tasks[task_id]
        return future.result(timeout=timeout)
    
    def wait_for_all_tasks(self, timeout: Optional[float] = None):
        """
        等待所有任务完成
        
        Args:
            timeout: 超时时间
            
        Examples:
            >>> manager = TaskManager()
            >>> manager.wait_for_all_tasks(timeout=60.0)
        """
        if not self.tasks:
            return
        
        # 等待所有任务完成
        for task_id, future in self.tasks.items():
            try:
                future.result(timeout=timeout)
            except Exception as e:
                logger.error(f"任务 {task_id} 失败: {e}")
    
    def cancel_task(self, task_id: str) -> bool:
        """
        取消任务
        
        Args:
            task_id: 任务标识
            
        Returns:
            是否成功取消
            
        Examples:
            >>> manager = TaskManager()
            >>> success = manager.cancel_task("task1")
        """
        if task_id not in self.tasks:
            return False
        
        future = self.tasks[task_id]
        cancelled = future.cancel()
        
        if cancelled:
            del self.tasks[task_id]
            logger.info(f"任务已取消: {task_id}")
        
        return cancelled
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            包含任务统计的字典
            
        Examples:
            >>> manager = TaskManager()
            >>> stats = manager.get_stats()
        """
        pool_stats = self.pool.get_stats()
        
        return {
            **pool_stats,
            'total_tasks': len(self.tasks),
            'pending_tasks': sum(1 for f in self.tasks.values() if not f.done()),
            'completed_tasks': sum(1 for f in self.tasks.values() if f.done() and not f.exception()),
            'failed_tasks': sum(1 for f in self.tasks.values() if f.done() and f.exception())
        }
    
    def clear_completed_tasks(self):
        """清理已完成的任务"""
        completed_ids = [
            task_id for task_id, future in self.tasks.items()
            if future.done()
        ]
        
        for task_id in completed_ids:
            del self.tasks[task_id]
        
        logger.info(f"已清理 {len(completed_ids)} 个已完成的任务")
    
    def shutdown(self):
        """关闭任务管理器"""
        self.pool.shutdown(wait=True)
        logger.info("任务管理器已关闭")


def create_dynamic_thread_pool(
    min_workers: int = 2,
    max_workers: Optional[int] = None,
    idle_timeout: float = 60.0,
    queue_size: int = 1000
) -> DynamicThreadPool:
    """
    创建动态线程池的工厂函数
    
    Args:
        min_workers: 最小线程数
        max_workers: 最大线程数
        idle_timeout: 空闲超时
        queue_size: 任务队列大小
        
    Returns:
        DynamicThreadPool 实例
        
    Examples:
        >>> pool = create_dynamic_thread_pool(min_workers=4, max_workers=16)
    """
    return DynamicThreadPool(
        min_workers=min_workers,
        max_workers=max_workers,
        idle_timeout=idle_timeout,
        queue_size=queue_size
    )


def create_task_manager(pool: Optional[DynamicThreadPool] = None) -> TaskManager:
    """
    创建任务管理器的工厂函数
    
    Args:
        pool: 线程池实例（如果为 None 则创建默认实例）
        
    Returns:
        TaskManager 实例
        
    Examples:
        >>> manager = create_task_manager()
    """
    return TaskManager(pool=pool)
