"""
动态并发管理器 - 智能调整工作线程数
根据系统负载、队列状态和处理速度动态调整并发数
"""

import os
import time
import threading
import logging
from typing import Optional, Dict, Any
from collections import deque
import psutil

logger = logging.getLogger(__name__)


class DynamicConcurrencyManager:
    """动态并发管理器 - 根据负载自动调整工作线程数"""
    
    def __init__(
        self,
        min_workers: int = 2,
        max_workers: int = 16,
        workers_per_cpu_core: float = 1.5,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.3,
        monitoring_interval: int = 10,
        enable_monitoring: bool = True
    ):
        """
        初始化动态并发管理器
        
        Args:
            min_workers: 最小工作线程数
            max_workers: 最大工作线程数
            workers_per_cpu_core: 每个CPU核心分配的工作线程数(可以是浮点数)
            scale_up_threshold: 负载超过此阈值时扩容
            scale_down_threshold: 负载低于此阈值时缩容
            monitoring_interval: 监控间隔(秒)
            enable_monitoring: 是否启用性能监控
        """
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.workers_per_cpu_core = workers_per_cpu_core
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.monitoring_interval = monitoring_interval
        self.enable_monitoring = enable_monitoring
        
        # 初始工作线程数
        cpu_count = os.cpu_count() or 4
        self.current_workers = min(
            max_workers,
            max(min_workers, int(cpu_count * workers_per_cpu_core))
        )
        
        # 性能指标
        self.metrics = {
            "queue_size": 0,
            "processing_rate": 0.0,  # 每秒处理的任务数
            "cpu_usage": 0.0,
            "memory_usage_mb": 0.0,
            "average_task_time": 0.0,
            "total_tasks_processed": 0,
            "last_adjustment_time": time.time()
        }
        
        # 任务处理时间记录(滑动窗口,最近100个任务)
        self.task_times = deque(maxlen=100)
        
        # 监控线程
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring = threading.Event()
        
        if enable_monitoring:
            self.start_monitoring()
        
        logger.info(f"初始化并发管理器: current_workers={self.current_workers}, min={min_workers}, max={max_workers}")
    
    def calculate_optimal_workers(self, queue_size: int, avg_task_time: float) -> int:
        """
        计算最优工作线程数
        
        Args:
            queue_size: 当前队列大小
            avg_task_time: 平均任务处理时间(秒)
        
        Returns:
            建议的工作线程数
        """
        # 基于CPU核心数的基准
        cpu_count = os.cpu_count() or 4
        base_workers = int(cpu_count * self.workers_per_cpu_core)
        
        # 基于队列积压情况调整
        if queue_size > 0 and avg_task_time > 0:
            # 估算需要多少线程才能在合理时间内清空队列
            # 假设目标是在60秒内处理完队列
            target_clearance_time = 60.0
            needed_workers = int((queue_size * avg_task_time) / target_clearance_time) + 1
            
            # 结合基准和需求
            optimal = max(base_workers, needed_workers)
        else:
            optimal = base_workers
        
        # 应用限制
        optimal = max(self.min_workers, min(self.max_workers, optimal))
        
        # 基于CPU和内存使用率的限制
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory_info = psutil.virtual_memory()
        
        # 如果CPU或内存使用率过高,限制线程数
        if cpu_usage > 85.0:
            optimal = max(self.min_workers, optimal - 2)
            logger.warning(f"CPU使用率过高({cpu_usage:.1f}%),降低并发数到 {optimal}")
        
        if memory_info.percent > 85.0:
            optimal = max(self.min_workers, optimal - 2)
            logger.warning(f"内存使用率过高({memory_info.percent:.1f}%),降低并发数到 {optimal}")
        
        return optimal
    
    def should_scale_up(self, queue_size: int, current_load: float) -> bool:
        """判断是否应该扩容"""
        if self.current_workers >= self.max_workers:
            return False
        
        # 条件1: 队列积压严重
        if queue_size > 50:
            return True
        
        # 条件2: 负载过高
        if current_load > self.scale_up_threshold:
            return True
        
        return False
    
    def should_scale_down(self, queue_size: int, current_load: float) -> bool:
        """判断是否应该缩容"""
        if self.current_workers <= self.min_workers:
            return False
        
        # 条件1: 队列几乎为空
        if queue_size < 10:
            # 条件2: 负载很低
            if current_load < self.scale_down_threshold:
                return True
        
        return False
    
    def adjust_workers(self, queue_size: int) -> int:
        """
        根据当前状态调整工作线程数
        
        Args:
            queue_size: 当前队列大小
        
        Returns:
            调整后的工作线程数
        """
        # 计算平均任务处理时间
        avg_task_time = sum(self.task_times) / len(self.task_times) if self.task_times else 1.0
        
        # 计算当前负载(基于队列大小和处理速度)
        processing_capacity = self.current_workers / max(avg_task_time, 0.1)
        current_load = queue_size / max(processing_capacity, 1.0)
        
        # 更新指标
        self.metrics["queue_size"] = queue_size
        self.metrics["average_task_time"] = avg_task_time
        self.metrics["processing_rate"] = processing_capacity
        
        # 计算最优工作线程数
        optimal_workers = self.calculate_optimal_workers(queue_size, avg_task_time)
        
        # 决策逻辑
        new_workers = self.current_workers
        
        if self.should_scale_up(queue_size, current_load):
            # 扩容:每次增加25%或至少2个
            increment = max(2, int(self.current_workers * 0.25))
            new_workers = min(self.max_workers, self.current_workers + increment)
            logger.info(f"🔼 扩容: {self.current_workers} -> {new_workers} (队列: {queue_size}, 负载: {current_load:.2f})")
        
        elif self.should_scale_down(queue_size, current_load):
            # 缩容:每次减少25%或至少1个
            decrement = max(1, int(self.current_workers * 0.25))
            new_workers = max(self.min_workers, self.current_workers - decrement)
            logger.info(f"🔽 缩容: {self.current_workers} -> {new_workers} (队列: {queue_size}, 负载: {current_load:.2f})")
        
        # 更新当前工作线程数
        if new_workers != self.current_workers:
            self.current_workers = new_workers
            self.metrics["last_adjustment_time"] = time.time()
        
        return self.current_workers
    
    def record_task_completion(self, task_duration: float):
        """
        记录任务完成情况
        
        Args:
            task_duration: 任务处理时间(秒)
        """
        self.task_times.append(task_duration)
        self.metrics["total_tasks_processed"] += 1
    
    def get_current_workers(self) -> int:
        """获取当前建议的工作线程数"""
        return self.current_workers
    
    def start_monitoring(self):
        """启动性能监控线程"""
        if self.monitoring_thread is not None:
            return
        
        def monitor_loop():
            while not self.stop_monitoring.is_set():
                try:
                    # 收集系统指标
                    self.metrics["cpu_usage"] = psutil.cpu_percent(interval=1.0)
                    
                    memory_info = psutil.virtual_memory()
                    self.metrics["memory_usage_mb"] = memory_info.used / (1024 * 1024)
                    
                    # 记录日志
                    if self.metrics["total_tasks_processed"] > 0:
                        logger.debug(
                            f"📊 性能指标: "
                            f"Workers={self.current_workers}, "
                            f"Queue={self.metrics['queue_size']}, "
                            f"CPU={self.metrics['cpu_usage']:.1f}%, "
                            f"Mem={self.metrics['memory_usage_mb']:.1f}MB, "
                            f"AvgTaskTime={self.metrics['average_task_time']:.2f}s, "
                            f"TotalTasks={self.metrics['total_tasks_processed']}"
                        )
                    
                    time.sleep(self.monitoring_interval)
                    
                except Exception as e:
                    logger.error(f"监控线程出错: {e}")
        
        self.monitoring_thread = threading.Thread(target=monitor_loop, daemon=True, name="ConcurrencyMonitor")
        self.monitoring_thread.start()
        logger.info("性能监控线程已启动")
    
    def stop_monitoring_thread(self):
        """停止监控线程"""
        if self.monitoring_thread:
            self.stop_monitoring.set()
            self.monitoring_thread.join(timeout=5.0)
            logger.info("性能监控线程已停止")
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return self.metrics.copy()
    
    def __del__(self):
        """析构函数,确保监控线程被停止"""
        self.stop_monitoring_thread()
