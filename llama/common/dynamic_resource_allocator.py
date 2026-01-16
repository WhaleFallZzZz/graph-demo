"""
动态资源分配器 - 跨Worker动态调整资源分配

当只有一个Worker活跃时，自动将其他空闲Worker的资源分配给活跃Worker，
实现资源的最优利用和性能提升。

核心功能：
- 监控所有Worker的活动状态
- 动态调整并发请求数、RPM限制、TPM限制
- 支持多种共享状态存储方式（Redis/文件）
- 与现有频率控制系统无缝集成

使用示例：
    from llama.common.dynamic_resource_allocator import DynamicScalingManager, ResourceAllocation
    
    # 创建基础资源分配配置
    base_allocation = ResourceAllocation(
        max_concurrent_requests=5,
        rpm_limit=60,
        tpm_limit=100000,
        num_workers=3
    )
    
    # 创建动态缩放管理器
    scaling_manager = DynamicScalingManager(
        worker_id='worker_1',
        total_workers=4,
        base_allocation=base_allocation,
        enable_scaling=True
    )
    
    # 设置资源分配回调函数
    def apply_allocation(allocation):
        print(f"应用资源分配: {allocation.to_dict()}")
    
    scaling_manager.set_allocation_callback(apply_allocation)
    
    # 启动动态缩放
    scaling_manager.start()
    
    # 更新Worker活动状态
    scaling_manager.update_activity(is_active=True, current_load=0.5, active_tasks=2)
    
    # 获取当前资源分配
    current_allocation = scaling_manager.get_current_allocation()
    
    # 停止动态缩放
    scaling_manager.stop()
"""

import os
import time
import json
import threading
import logging
import fcntl
import uuid
from typing import Dict, Optional, List, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class WorkerState:
    """
    Worker状态信息
    
    存储单个Worker的运行时状态信息，包括活动状态、负载、任务数等。
    
    Attributes:
        worker_id: Worker的唯一标识符
        is_active: Worker是否处于活跃状态
        last_active_time: 最后一次活动的时间戳（Unix时间戳）
        current_load: 当前负载（0.0-1.0），表示Worker的繁忙程度
        active_tasks: 当前活跃的任务数量
        allocated_resources: 已分配的资源字典
    """
    worker_id: str
    is_active: bool
    last_active_time: float
    current_load: float
    active_tasks: int
    allocated_resources: Dict[str, int]


@dataclass
class ResourceAllocation:
    """
    资源分配配置
    
    定义系统可用的资源限制，包括并发请求数、RPM（每分钟请求数）、TPM（每分钟Token数）等。
    
    Attributes:
        max_concurrent_requests: 最大并发请求数
        rpm_limit: RPM（每分钟请求数）限制
        tpm_limit: TPM（每分钟Token数）限制
        num_workers: 工作线程数量
    """
    max_concurrent_requests: int
    rpm_limit: int
    tpm_limit: int
    num_workers: int
    
    def to_dict(self) -> Dict[str, int]:
        """
        转换为字典格式
        
        Returns:
            包含所有资源分配配置的字典
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> 'ResourceAllocation':
        """
        从字典创建ResourceAllocation实例
        
        Args:
            data: 包含资源分配配置的字典
            
        Returns:
            ResourceAllocation实例
        """
        return cls(**data)


class WorkerActivityMonitor:
    """
    Worker活动监控器 - 跟踪所有Worker的活动状态
    
    负责监控系统中所有Worker的活动状态，通过共享存储（文件或Redis）同步状态信息。
    定期清理过期的Worker状态，并提供活跃Worker列表查询功能。
    
    主要功能：
    - 跟踪本地Worker的活动状态
    - 与其他Worker同步状态信息
    - 自动清理过期的Worker状态
    - 提供活跃Worker列表查询
    """
    
    def __init__(
        self,
        worker_id: str,
        total_workers: int = 4,
        activity_timeout: float = 30.0,
        monitoring_interval: float = 5.0,
        storage_backend: str = "file"
    ):
        """
        初始化Worker活动监控器
        
        Args:
            worker_id: 当前Worker的唯一标识
            total_workers: 总Worker数量
            activity_timeout: 活动超时时间（秒），超过此时间未活动则视为空闲
            monitoring_interval: 监控间隔（秒），多久同步一次状态
            storage_backend: 存储后端（"file" 或 "redis"）
        """
        self.worker_id = worker_id
        self.total_workers = total_workers
        self.activity_timeout = activity_timeout
        self.monitoring_interval = monitoring_interval
        self.storage_backend = storage_backend
        
        # 共享状态存储路径
        self.state_file = Path(os.getcwd()) / ".worker_activity_state.json"
        
        # 本地状态
        self.local_state: WorkerState = WorkerState(
            worker_id=worker_id,
            is_active=False,
            last_active_time=time.time(),
            current_load=0.0,
            active_tasks=0,
            allocated_resources={}
        )
        
        # 监控线程
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_monitoring = threading.Event()
        
        # 状态锁，保护本地状态的并发访问
        self.state_lock = threading.Lock()
        
        logger.info(f"初始化Worker活动监控器: worker_id={worker_id}, total_workers={total_workers}")
    
    def _get_all_worker_states(self) -> Dict[str, WorkerState]:
        """
        获取所有Worker的状态
        
        从共享存储中读取所有Worker的状态信息。
        
        Returns:
            Worker ID到WorkerState的映射字典
        """
        if self.storage_backend == "file":
            return self._read_from_file()
        elif self.storage_backend == "redis":
            return self._read_from_redis()
        else:
            logger.warning(f"未知的存储后端: {self.storage_backend}，使用本地状态")
            return {self.worker_id: self.local_state}
    
    def _read_from_file(self) -> Dict[str, WorkerState]:
        """
        从文件读取所有Worker状态
        
        使用文件锁确保读取一致性，支持重试机制。
        
        Returns:
            Worker ID到WorkerState的映射字典
        """
        max_retries = 3
        retry_delay = 0.1
        
        for attempt in range(max_retries):
            try:
                if not self.state_file.exists():
                    return {}
                
                # 使用文件锁确保读取一致性
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    # 尝试获取共享锁（读取锁）
                    try:
                        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                    except (AttributeError, OSError):
                        pass
                    
                    try:
                        data = json.load(f)
                    finally:
                        try:
                            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                        except (AttributeError, OSError):
                            pass
                
                states = {}
                for worker_id, state_data in data.items():
                    states[worker_id] = WorkerState(**state_data)
                
                return states
                
            except json.JSONDecodeError as e:
                logger.warning(f"解析Worker状态文件失败（尝试 {attempt + 1}/{max_retries}）: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    logger.error(f"解析Worker状态文件最终失败: {e}")
                    return {}
            except Exception as e:
                logger.warning(f"读取Worker状态文件失败（尝试 {attempt + 1}/{max_retries}）: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    logger.error(f"读取Worker状态文件最终失败: {e}")
                    return {}
    
    def _write_to_file(self, states: Dict[str, WorkerState]):
        """
        写入所有Worker状态到文件
        
        使用原子写入操作（先写临时文件，再替换原文件）确保数据一致性。
        使用文件锁确保写入的并发安全。
        
        Args:
            states: Worker ID到WorkerState的映射字典
        """
        max_retries = 3
        retry_delay = 0.1
        
        for attempt in range(max_retries):
            try:
                data = {}
                for worker_id, state in states.items():
                    data[worker_id] = asdict(state)
                
                # 确保父目录存在
                self.state_file.parent.mkdir(parents=True, exist_ok=True)
                
                # 使用唯一的临时文件名（包含进程ID和Worker ID）
                temp_file = self.state_file.with_suffix(
                    f'.tmp.{os.getpid()}.{self.worker_id}.{uuid.uuid4().hex[:8]}'
                )
                
                # 写入临时文件
                with open(temp_file, 'w', encoding='utf-8') as f:
                    # 尝试获取排他锁（写入锁）
                    try:
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    except (AttributeError, OSError):
                        pass
                    
                    try:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                        f.flush()
                        os.fsync(f.fileno())
                    finally:
                        try:
                            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                        except (AttributeError, OSError):
                            pass
                
                # 原子替换
                temp_file.replace(self.state_file)
                
                return
                
            except Exception as e:
                logger.warning(f"写入Worker状态文件失败（尝试 {attempt + 1}/{max_retries}）: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    logger.error(f"写入Worker状态文件最终失败: {e}")
    
    def _read_from_redis(self) -> Dict[str, WorkerState]:
        """
        从Redis读取所有Worker状态
        
        从Redis哈希表中读取所有Worker的状态信息。
        
        Returns:
            Worker ID到WorkerState的映射字典
        """
        try:
            import redis
            redis_client = redis.Redis(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', 6379)),
                db=int(os.getenv('REDIS_DB', 0)),
                decode_responses=True
            )
            
            key = "worker_activity_states"
            data = redis_client.hgetall(key)
            
            states = {}
            for worker_id, state_json in data.items():
                state_data = json.loads(state_json)
                states[worker_id] = WorkerState(**state_data)
            
            return states
        except ImportError:
            logger.warning("Redis未安装，回退到文件存储")
            self.storage_backend = "file"
            return self._read_from_file()
        except Exception as e:
            logger.error(f"从Redis读取Worker状态失败: {e}")
            return {}
    
    def _write_to_redis(self, states: Dict[str, WorkerState]):
        """
        写入所有Worker状态到Redis
        
        将所有Worker的状态信息写入Redis哈希表，并设置过期时间。
        
        Args:
            states: Worker ID到WorkerState的映射字典
        """
        try:
            import redis
            redis_client = redis.Redis(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', 6379)),
                db=int(os.getenv('REDIS_DB', 0)),
                decode_responses=True
            )
            
            key = "worker_activity_states"
            pipe = redis_client.pipeline()
            
            for worker_id, state in states.items():
                state_json = json.dumps(asdict(state), ensure_ascii=False)
                pipe.hset(key, worker_id, state_json)
            
            pipe.expire(key, 3600)
            pipe.execute()
            
        except Exception as e:
            logger.error(f"写入Redis Worker状态失败: {e}")
    
    def update_local_state(
        self,
        is_active: bool,
        current_load: float = 0.0,
        active_tasks: int = 0
    ):
        """
        更新本地Worker状态
        
        更新当前Worker的状态信息，包括活动状态、负载和任务数。
        如果Worker处于活跃状态，会自动更新最后活动时间。
        
        Args:
            is_active: 是否活跃
            current_load: 当前负载（0.0-1.0）
            active_tasks: 活跃任务数
        """
        with self.state_lock:
            self.local_state.is_active = is_active
            self.local_state.current_load = current_load
            self.local_state.active_tasks = active_tasks
            
            if is_active:
                self.local_state.last_active_time = time.time()
            
            logger.debug(
                f"更新本地Worker状态: {self.worker_id}, "
                f"active={is_active}, load={current_load:.2f}, tasks={active_tasks}"
            )
    
    def sync_state(self):
        """
        同步本地状态到共享存储
        
        将本地Worker的状态同步到共享存储（文件或Redis），
        同时清理过期的Worker状态。
        """
        with self.state_lock:
            all_states = self._get_all_worker_states()
            all_states[self.worker_id] = self.local_state
            
            # 清理过期的Worker状态
            current_time = time.time()
            expired_workers = []
            for worker_id, state in all_states.items():
                if current_time - state.last_active_time > self.activity_timeout * 2:
                    expired_workers.append(worker_id)
            
            for worker_id in expired_workers:
                del all_states[worker_id]
                logger.debug(f"清理过期Worker状态: {worker_id}")
            
            # 写入共享存储
            if self.storage_backend == "file":
                self._write_to_file(all_states)
            elif self.storage_backend == "redis":
                self._write_to_redis(all_states)
    
    def get_active_workers(self) -> List[str]:
        """
        获取当前活跃的Worker列表
        
        Returns:
            活跃Worker ID列表
        """
        all_states = self._get_all_worker_states()
        current_time = time.time()
        
        active_workers = []
        for worker_id, state in all_states.items():
            if state.is_active and (current_time - state.last_active_time) < self.activity_timeout:
                active_workers.append(worker_id)
        
        return active_workers
    
    def get_active_worker_count(self) -> int:
        """
        获取活跃Worker数量
        
        Returns:
            活跃Worker数量
        """
        return len(self.get_active_workers())
    
    def start_monitoring(self):
        """
        启动监控线程
        
        启动后台监控线程，定期同步状态并记录活跃Worker数量。
        """
        if self.monitor_thread is not None:
            return
        
        def monitor_loop():
            while not self.stop_monitoring.is_set():
                try:
                    # 同步状态
                    self.sync_state()
                    
                    # 记录活跃Worker数量
                    active_count = self.get_active_worker_count()
                    logger.debug(f"当前活跃Worker数量: {active_count}/{self.total_workers}")
                    
                    # 等待下一次监控
                    self.stop_monitoring.wait(self.monitoring_interval)
                    
                except Exception as e:
                    logger.error(f"监控线程出错: {e}")
        
        self.monitor_thread = threading.Thread(
            target=monitor_loop,
            daemon=True,
            name=f"WorkerMonitor-{self.worker_id}"
        )
        self.monitor_thread.start()
        
        logger.info(f"Worker监控线程已启动: {self.worker_id}")
    
    def stop_monitoring_thread(self):
        """
        停止监控线程
        
        停止后台监控线程，等待线程退出。
        """
        if self.monitor_thread:
            self.stop_monitoring.set()
            self.monitor_thread.join(timeout=5.0)
            logger.info(f"Worker监控线程已停止: {self.worker_id}")
    
    def cleanup(self):
        """
        清理资源
        
        停止监控线程，并从共享存储中移除本Worker的状态。
        """
        self.stop_monitoring_thread()
        
        # 从共享存储中移除本Worker的状态
        try:
            all_states = self._get_all_worker_states()
            if self.worker_id in all_states:
                del all_states[self.worker_id]
                
                if self.storage_backend == "file":
                    self._write_to_file(all_states)
                elif self.storage_backend == "redis":
                    self._write_to_redis(all_states)
                
                logger.info(f"已清理Worker状态: {self.worker_id}")
        except Exception as e:
            logger.error(f"清理Worker状态失败: {e}")


class DynamicResourceAllocator:
    """
    动态资源分配器 - 根据活跃Worker数量动态调整资源分配
    
    根据当前活跃的Worker数量，动态调整每个Worker可用的资源限制。
    当只有一个Worker活跃时，将所有资源分配给该Worker，实现性能最大化。
    当多个Worker活跃时，平均分配资源。
    
    主要功能：
    - 监控活跃Worker数量
    - 动态计算最优资源分配
    - 通过回调函数应用资源分配
    - 提供资源分配状态查询
    """
    
    def __init__(
        self,
        worker_id: str,
        total_workers: int = 4,
        base_allocation: Optional[ResourceAllocation] = None,
        monitor: Optional[WorkerActivityMonitor] = None,
        adjustment_interval: float = 10.0,
        enable_scaling: bool = True
    ):
        """
        初始化动态资源分配器
        
        Args:
            worker_id: 当前Worker的唯一标识
            total_workers: 总Worker数量
            base_allocation: 基础资源分配（每个Worker的默认资源）
            monitor: Worker活动监控器，如果为None则自动创建
            adjustment_interval: 资源调整间隔（秒）
            enable_scaling: 是否启用动态缩放
        """
        self.worker_id = worker_id
        self.total_workers = total_workers
        self.adjustment_interval = adjustment_interval
        self.enable_scaling = enable_scaling
        
        # 基础资源分配（每个Worker的默认资源）
        if base_allocation is None:
            base_allocation = ResourceAllocation(
                max_concurrent_requests=10,
                rpm_limit=200,
                tpm_limit=10000,
                num_workers=10
            )
        
        self.base_allocation = base_allocation
        
        # 总资源（所有Worker共享）
        self.total_resources = ResourceAllocation(
            max_concurrent_requests=base_allocation.max_concurrent_requests * total_workers,
            rpm_limit=base_allocation.rpm_limit * total_workers,
            tpm_limit=base_allocation.tpm_limit * total_workers,
            num_workers=base_allocation.num_workers * total_workers
        )
        
        # 当前分配的资源
        self.current_allocation = ResourceAllocation(**asdict(base_allocation))
        
        # Worker活动监控器
        if monitor is None:
            monitor = WorkerActivityMonitor(
                worker_id=worker_id,
                total_workers=total_workers
            )
        self.monitor = monitor
        
        # 调整线程
        self.adjustment_thread: Optional[threading.Thread] = None
        self.stop_adjustment = threading.Event()
        
        # 资源分配回调函数
        self.allocation_callback: Optional[Callable[[ResourceAllocation], None]] = None
        
        logger.info(
            f"初始化动态资源分配器: worker_id={worker_id}, "
            f"base_allocation={base_allocation.to_dict()}, "
            f"total_resources={self.total_resources.to_dict()}"
        )
    
    def set_allocation_callback(self, callback: Callable[[ResourceAllocation], None]):
        """
        设置资源分配回调函数
        
        当资源分配发生变化时，会调用此回调函数。
        
        Args:
            callback: 回调函数，接收ResourceAllocation参数
        """
        self.allocation_callback = callback
        logger.info("资源分配回调函数已设置")
    
    def calculate_optimal_allocation(self, active_workers: List[str]) -> ResourceAllocation:
        """
        计算最优资源分配
        
        根据活跃Worker数量计算最优的资源分配策略：
        - 0个活跃Worker：使用基础分配
        - 1个活跃Worker：分配所有资源（性能最大化）
        - 多个活跃Worker：平均分配资源
        
        Args:
            active_workers: 活跃Worker列表
            
        Returns:
            最优资源分配
        """
        active_count = len(active_workers)
        
        if active_count == 0:
            return ResourceAllocation(**asdict(self.base_allocation))
        
        if active_count == 1:
            logger.info(f"🚀 激活动态缩放: 只有1个活跃Worker，分配全部资源")
            return ResourceAllocation(**asdict(self.total_resources))
        
        # 多个活跃Worker，平均分配资源
        avg_concurrent = self.total_resources.max_concurrent_requests // active_count
        avg_rpm = self.total_resources.rpm_limit // active_count
        avg_tpm = self.total_resources.tpm_limit // active_count
        avg_workers = self.total_resources.num_workers // active_count
        
        allocation = ResourceAllocation(
            max_concurrent_requests=avg_concurrent,
            rpm_limit=avg_rpm,
            tpm_limit=avg_tpm,
            num_workers=avg_workers
        )
        
        logger.info(
            f"📊 平均分配资源: {active_count}个活跃Worker, "
            f"concurrent={avg_concurrent}, rpm={avg_rpm}, tpm={avg_tpm}, workers={avg_workers}"
        )
        
        return allocation
    
    def adjust_resources(self):
        """
        调整资源分配
        
        根据当前活跃Worker数量，计算最优资源分配并应用。
        如果资源分配发生变化，会调用回调函数通知应用层。
        """
        if not self.enable_scaling:
            return
        
        # 获取活跃Worker列表
        active_workers = self.monitor.get_active_workers()
        
        # 计算最优分配
        optimal_allocation = self.calculate_optimal_allocation(active_workers)
        
        # 检查是否需要调整
        if optimal_allocation.to_dict() != self.current_allocation.to_dict():
            old_allocation = self.current_allocation.to_dict()
            new_allocation = optimal_allocation.to_dict()
            
            self.current_allocation = optimal_allocation
            
            logger.info(
                f"🔄 资源分配调整: {self.worker_id}\n"
                f"  旧配置: {old_allocation}\n"
                f"  新配置: {new_allocation}\n"
                f"  活跃Worker: {len(active_workers)}/{self.total_workers}"
            )
            
            # 调用回调函数
            if self.allocation_callback:
                try:
                    self.allocation_callback(self.current_allocation)
                except Exception as e:
                    logger.error(f"资源分配回调函数执行失败: {e}")
    
    def start_adjustment(self):
        """
        启动资源调整线程
        
        启动后台调整线程，定期检查并调整资源分配。
        """
        if self.adjustment_thread is not None:
            return
        
        def adjustment_loop():
            while not self.stop_adjustment.is_set():
                try:
                    # 调整资源
                    self.adjust_resources()
                    
                    # 等待下一次调整
                    self.stop_adjustment.wait(self.adjustment_interval)
                    
                except Exception as e:
                    logger.error(f"资源调整线程出错: {e}")
        
        self.adjustment_thread = threading.Thread(
            target=adjustment_loop,
            daemon=True,
            name=f"ResourceAdjuster-{self.worker_id}"
        )
        self.adjustment_thread.start()
        
        logger.info(f"资源调整线程已启动: {self.worker_id}")
    
    def stop_adjustment_thread(self):
        """
        停止资源调整线程
        
        停止后台调整线程，等待线程退出。
        """
        if self.adjustment_thread:
            self.stop_adjustment.set()
            self.adjustment_thread.join(timeout=5.0)
            logger.info(f"资源调整线程已停止: {self.worker_id}")
    
    def get_current_allocation(self) -> ResourceAllocation:
        """
        获取当前资源分配
        
        Returns:
            当前资源分配配置
        """
        return self.current_allocation
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """
        获取缩放状态
        
        Returns:
            包含缩放状态信息的字典，包括：
            - worker_id: Worker ID
            - total_workers: 总Worker数量
            - active_workers: 活跃Worker数量
            - active_worker_ids: 活跃Worker ID列表
            - is_scaling_enabled: 是否启用缩放
            - base_allocation: 基础资源分配
            - current_allocation: 当前资源分配
            - total_resources: 总资源
            - utilization_ratio: 资源利用率
        """
        active_workers = self.monitor.get_active_workers()
        active_count = len(active_workers)
        
        return {
            'worker_id': self.worker_id,
            'total_workers': self.total_workers,
            'active_workers': active_count,
            'active_worker_ids': active_workers,
            'is_scaling_enabled': self.enable_scaling,
            'base_allocation': self.base_allocation.to_dict(),
            'current_allocation': self.current_allocation.to_dict(),
            'total_resources': self.total_resources.to_dict(),
            'utilization_ratio': {
                'concurrent': self.current_allocation.max_concurrent_requests / self.base_allocation.max_concurrent_requests,
                'rpm': self.current_allocation.rpm_limit / self.base_allocation.rpm_limit,
                'tpm': self.current_allocation.tpm_limit / self.base_allocation.tpm_limit,
                'workers': self.current_allocation.num_workers / self.base_allocation.num_workers
            }
        }
    
    def cleanup(self):
        """
        清理资源
        
        停止调整线程和监控线程，释放所有资源。
        """
        self.stop_adjustment_thread()
        self.monitor.cleanup()
        logger.info(f"动态资源分配器已清理: {self.worker_id}")


class DynamicScalingManager:
    """
    动态缩放管理器 - 统一管理Worker监控和资源分配
    
    提供统一的接口来管理动态资源分配功能，简化使用。
    内部集成了WorkerActivityMonitor和DynamicResourceAllocator。
    
    主要功能：
    - 统一管理Worker监控和资源分配
    - 提供简洁的API接口
    - 支持上下文管理器协议
    - 提供状态查询功能
    """
    
    def __init__(
        self,
        worker_id: str,
        total_workers: int = 4,
        base_allocation: Optional[ResourceAllocation] = None,
        enable_scaling: bool = True
    ):
        """
        初始化动态缩放管理器
        
        Args:
            worker_id: 当前Worker的唯一标识
            total_workers: 总Worker数量
            base_allocation: 基础资源分配
            enable_scaling: 是否启用动态缩放
        """
        self.worker_id = worker_id
        
        # 创建Worker活动监控器
        self.monitor = WorkerActivityMonitor(
            worker_id=worker_id,
            total_workers=total_workers
        )
        
        # 创建动态资源分配器
        self.allocator = DynamicResourceAllocator(
            worker_id=worker_id,
            total_workers=total_workers,
            base_allocation=base_allocation,
            monitor=self.monitor,
            enable_scaling=enable_scaling
        )
        
        logger.info(f"动态缩放管理器已初始化: {worker_id}")
    
    def start(self):
        """
        启动动态缩放
        
        启动Worker监控和资源调整功能。
        """
        self.monitor.start_monitoring()
        self.allocator.start_adjustment()
        logger.info(f"动态缩放已启动: {self.worker_id}")
    
    def stop(self):
        """
        停止动态缩放
        
        停止Worker监控和资源调整功能，释放所有资源。
        """
        self.allocator.cleanup()
        self.monitor.cleanup()
        logger.info(f"动态缩放已停止: {self.worker_id}")
    
    def update_activity(
        self,
        is_active: bool,
        current_load: float = 0.0,
        active_tasks: int = 0
    ):
        """
        更新Worker活动状态
        
        Args:
            is_active: 是否活跃
            current_load: 当前负载（0.0-1.0）
            active_tasks: 活跃任务数
        """
        self.monitor.update_local_state(is_active, current_load, active_tasks)
    
    def set_allocation_callback(self, callback: Callable[[ResourceAllocation], None]):
        """
        设置资源分配回调函数
        
        Args:
            callback: 回调函数，接收ResourceAllocation参数
        """
        self.allocator.set_allocation_callback(callback)
    
    def get_current_allocation(self) -> ResourceAllocation:
        """
        获取当前资源分配
        
        Returns:
            当前资源分配配置
        """
        return self.allocator.get_current_allocation()
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取状态
        
        Returns:
            包含缩放状态信息的字典
        """
        return self.allocator.get_scaling_status()
    
    def __enter__(self):
        """
        支持上下文管理器协议
        
        进入上下文时自动启动动态缩放。
        
        Returns:
            DynamicScalingManager实例
        """
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        上下文管理器退出时停止
        
        退出上下文时自动停止动态缩放。
        """
        self.stop()
