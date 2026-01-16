#!/usr/bin/env python3
"""
SSE 进度推送模块

提供知识图谱构建过程的实时进度推送功能，基于 Server-Sent Events (SSE) 协议。

主要功能：
- 实时进度推送：通过 SSE 向客户端推送知识图谱构建进度
- 进度缓存：缓存进度信息，支持客户端重连后获取最新进度
- 多客户端支持：支持多个客户端同时监听进度
- 进度追踪：提供进度追踪器，自动计算进度百分比和剩余时间
- 事件类型：支持进度事件、错误事件、完成事件等

使用示例：
    from llama.progress_sse import ProgressTracker, progress_manager
    
    # 方式1：使用 ProgressTracker（推荐用于有明确步骤的任务）
    tracker = ProgressTracker(client_id='client-123', total_steps=8)
    tracker.update_progress('initialization', '正在初始化...', 1)
    tracker.update_progress('document_loading', '正在加载文档...', 2)
    tracker.complete({'success': True})
    
    # 方式2：使用 progress_callback（推荐用于全局进度推送）
    progress_callback('knowledge_graph', '开始构建知识图谱...', 20)
    progress_callback('knowledge_graph', '知识图谱构建完成', 100)
    
    # 方式3：直接使用 progress_manager
    progress_manager.add_listener('client-123', callback_function)
    progress_manager.send_progress('client-123', {'type': 'progress', 'message': '...'})
"""

import json
import queue
from typing import Dict, Any, Optional, Callable
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ProgressSSE:
    """
    SSE 进度推送管理器
    
    管理 SSE 进度推送的核心类，负责监听器管理、进度缓存和消息分发。
    
    主要功能：
    - 监听器管理：添加和移除客户端监听器
    - 进度缓存：缓存每个客户端的进度信息，支持 TTL 和最大缓存大小限制
    - 消息分发：向指定客户端或所有客户端发送进度消息
    - 缓存清理：自动清理过期的缓存条目
    
    缓存策略：
    - TTL（Time To Live）：缓存条目在指定时间后自动过期
    - 最大缓存大小：超过最大缓存大小时，删除最旧的条目
    - 惰性清理：在添加监听器时触发缓存清理
    
    Attributes:
        listeners (Dict[str, Callable]): 客户端 ID 到回调函数的映射
        progress_cache (Dict[str, Dict]): 客户端 ID 到进度信息的映射
        max_cache_size (int): 最大缓存条目数，默认为 1000
        ttl_seconds (int): 缓存条目的生存时间（秒），默认为 3600（1 小时）
    """
    
    def __init__(self, max_cache_size: int = 1000, ttl_seconds: int = 3600):
        """
        初始化 SSE 进度推送管理器
        
        Args:
            max_cache_size: 最大缓存条目数，默认为 1000
            ttl_seconds: 缓存条目的生存时间（秒），默认为 3600（1 小时）
        """
        self.listeners = {}
        self.progress_cache = {}
        self.max_cache_size = max_cache_size
        self.ttl_seconds = ttl_seconds
        
    def add_listener(self, client_id: str, callback: Callable):
        """
        添加监听器
        
        为指定客户端添加进度监听器，后续发送给该客户端的进度消息
        将通过回调函数传递。添加监听器时会触发缓存清理。
        
        Args:
            client_id: 客户端唯一标识符
            callback: 回调函数，接收进度数据字典作为参数
        """
        self._cleanup_cache()
        self.listeners[client_id] = callback
        logger.info(f"添加监听器: {client_id}")
        
    def remove_listener(self, client_id: str):
        """
        移除监听器
        
        移除指定客户端的监听器，后续将不再向该客户端发送进度消息。
        
        Args:
            client_id: 客户端唯一标识符
        """
        if client_id in self.listeners:
            del self.listeners[client_id]
            logger.info(f"移除监听器: {client_id}")
            
    def _cleanup_cache(self):
        """
        清理过期的缓存
        
        清理过期的缓存条目，确保缓存大小在限制范围内。
        清理策略：
        1. 删除超过 TTL 的条目
        2. 如果仍然超过最大缓存大小，删除最旧的条目
        
        此方法采用惰性清理策略，在添加监听器时调用。
        """
        try:
            now_ts = datetime.now().timestamp()
            
            # 1. 清理过期条目
            expired_keys = []
            for client_id, info in self.progress_cache.items():
                cached_ts = info.get('_ts', 0)
                if now_ts - cached_ts > self.ttl_seconds:
                    expired_keys.append(client_id)
            
            for key in expired_keys:
                del self.progress_cache[key]
                
            # 2. 如果仍然超过最大大小，清理最旧的
            if len(self.progress_cache) > self.max_cache_size:
                # 按时间戳排序
                sorted_items = sorted(
                    self.progress_cache.items(), 
                    key=lambda x: x[1].get('_ts', 0)
                )
                # 删除最旧的 (当前数量 - 目标数量) 个
                num_to_remove = len(self.progress_cache) - self.max_cache_size
                for i in range(num_to_remove):
                    del self.progress_cache[sorted_items[i][0]]
                    
                logger.info(f"清理缓存: 移除了 {len(expired_keys) + num_to_remove} 个条目")
        except Exception as e:
            logger.error(f"清理缓存失败: {e}")

    def send_progress(self, client_id: str, data: Dict[str, Any]):
        """
        发送进度信息给指定客户端
        
        向指定客户端发送进度消息，并缓存该消息以便客户端重连后获取。
        
        Args:
            client_id: 客户端唯一标识符
            data: 进度数据字典
        """
        if client_id in self.listeners:
            try:
                # 缓存进度信息
                self.progress_cache[client_id] = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    '_ts': datetime.now().timestamp(),
                    'data': data
                }
                
                # 调用回调函数发送SSE事件
                callback = self.listeners[client_id]
                callback(data)
                
            except Exception as e:
                logger.error(f"发送进度失败: {e}")
                
    def broadcast_progress(self, data: Dict[str, Any]):
        """
        广播进度信息给所有监听器
        
        向所有已注册的监听器发送相同的进度消息。
        
        Args:
            data: 进度数据字典
        """
        for client_id in list(self.listeners.keys()):
            self.send_progress(client_id, data)
            
    def get_progress(self, client_id: str) -> Optional[Dict[str, Any]]:
        """
        获取指定客户端的进度信息
        
        从缓存中获取指定客户端的最新进度信息。
        
        Args:
            client_id: 客户端唯一标识符
            
        Returns:
            客户端的进度信息字典，如果不存在则返回 None
        """
        return self.progress_cache.get(client_id)

# 全局进度管理器实例
progress_manager = ProgressSSE()


def sse_event(data: Dict[str, Any]) -> str:
    """
    生成 SSE 事件数据
    
    将数据字典转换为 SSE（Server-Sent Events）格式的字符串。
    SSE 格式：`data: {JSON数据}\n\n`
    
    Args:
        data: 要发送的数据字典
        
    Returns:
        SSE 格式的字符串
        
    Example:
        >>> sse_event({'type': 'progress', 'message': 'Processing...'})
        'data: {"type": "progress", "message": "Processing..."}\\n\\n'
    """
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_progress_event(
    stage: str, 
    message: str, 
    percentage: Optional[float] = None,
    details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    创建进度事件
    
    创建一个表示进度的标准事件字典。
    
    Args:
        stage: 阶段标识符，如 'initialization', 'document_loading', 'knowledge_graph' 等
        message: 进度消息描述
        percentage: 进度百分比（0-100），可选
        details: 额外的详细信息字典，可选
        
    Returns:
        进度事件字典，包含以下字段：
        - type: 固定为 'progress'
        - stage: 阶段标识符
        - message: 进度消息
        - timestamp: 时间戳（ISO 格式）
        - percentage: 进度百分比（如果提供）
        - details: 额外详细信息（如果提供）
    """
    event = {
        'type': 'progress',
        'stage': stage,
        'message': message,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    if percentage is not None:
        event['percentage'] = round(percentage, 1)
        
    if details:
        event['details'] = details
        
    return event


def create_error_event(stage: str, message: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    创建错误事件
    
    创建一个表示错误的标准事件字典。
    
    Args:
        stage: 发生错误的阶段标识符
        message: 错误消息描述
        details: 额外的详细信息字典，可选
        
    Returns:
        错误事件字典，包含以下字段：
        - type: 固定为 'error'
        - stage: 阶段标识符
        - message: 错误消息
        - timestamp: 时间戳（ISO 格式）
        - details: 额外详细信息（如果提供）
    """
    event = {
        'type': 'error',
        'stage': stage,
        'message': message,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    if details:
        event['details'] = details
        
    return event


def create_complete_event(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    创建完成事件
    
    创建一个表示任务完成的标准事件字典。
    
    Args:
        result: 任务结果数据字典
        
    Returns:
        完成事件字典，包含以下字段：
        - type: 固定为 'complete'
        - stage: 固定为 'complete'
        - message: 固定为 '处理完成'
        - result: 任务结果数据
        - timestamp: 时间戳（ISO 格式）
    """
    return {
        'type': 'complete',
        'stage': 'complete',
        'message': '处理完成',
        'result': result,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

class ProgressTracker:
    """
    进度追踪器
    
    用于追踪任务的进度，自动计算进度百分比和剩余时间预估。
    适用于有明确步骤数或需要精确进度控制的任务。
    
    主要功能：
    - 步骤追踪：追踪当前步骤和总步骤数
    - 进度计算：自动计算进度百分比
    - 时间预估：根据已用时间和进度百分比预估剩余时间
    - 阶段记录：记录每个阶段的时间和进度信息
    - 误差补偿：确保进度不回退（除非是完成阶段）
    
    使用示例：
        tracker = ProgressTracker(client_id='client-123', total_steps=8)
        
        # 方式1：指定步骤号
        tracker.update_progress('initialization', '正在初始化...', step=1)
        
        # 方式2：自动递增步骤
        tracker.update_progress('document_loading', '正在加载文档...')
        
        # 方式3：指定进度百分比
        tracker.update_progress('knowledge_graph', '开始构建...', percentage=50)
        
        # 发送错误
        tracker.error('knowledge_graph', '构建失败')
        
        # 完成任务
        tracker.complete({'success': True})
    
    Attributes:
        client_id (str): 客户端唯一标识符
        total_steps (int): 总步骤数
        current_step (int): 当前步骤
        start_time (datetime): 任务开始时间
        stage_times (Dict[str, Dict]): 各阶段的时间和进度记录
        last_percentage (float): 上一次的进度百分比（用于误差补偿）
    """
    
    def __init__(self, client_id: str, total_steps: int = 8):
        """
        初始化进度追踪器
        
        Args:
            client_id: 客户端唯一标识符
            total_steps: 总步骤数，默认为 8
        """
        self.client_id = client_id
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = datetime.now()
        self.stage_times = {}
        self.last_percentage = 0.0
        
    def update_progress(self, stage: str, message: str, step: Optional[int] = None, percentage: Optional[float] = None):
        """
        更新进度
        
        更新任务进度，自动计算进度百分比和剩余时间预估。
        支持三种更新方式：
        1. 指定步骤号：step 参数指定当前步骤
        2. 自动递增：不指定 step 和 percentage，自动递增步骤
        3. 指定百分比：percentage 参数直接指定进度百分比
        
        Args:
            stage: 阶段标识符，如 'initialization', 'document_loading' 等
            message: 进度消息描述
            step: 当前步骤号（可选）
            percentage: 进度百分比（可选，0-100）
        """
        if step is not None:
            self.current_step = step
        elif percentage is None:
            self.current_step += 1
            
        if percentage is None:
            percentage = (self.current_step / self.total_steps) * 100
            
        # 误差补偿机制：确保进度不回退
        if percentage < self.last_percentage:
            # 只有当阶段改变时才允许进度"重置"（虽然通常也不应该），或者我们忽略小的回退
            # 这里简单处理：取最大值，除非是完成阶段
            if stage != 'complete':
                percentage = self.last_percentage
        
        self.last_percentage = percentage
        
        # 计算剩余时间预估
        elapsed = (datetime.now() - self.start_time).total_seconds()
        remaining_seconds = 0
        if percentage > 0:
            estimated_total = elapsed / (percentage / 100)
            remaining_seconds = max(0, estimated_total - elapsed)
        
        # 记录阶段时间
        self.stage_times[stage] = {
            'start_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'step': self.current_step,
            'percentage': percentage
        }
        
        # 创建进度事件
        event = create_progress_event(
            stage=stage,
            message=message,
            percentage=percentage,
            details={
                'step': self.current_step,
                'total_steps': self.total_steps,
                'stage_times': self.stage_times,
                'elapsed_seconds': round(elapsed, 1),
                'remaining_seconds': round(remaining_seconds, 1)
            }
        )
        
        # 发送进度
        progress_manager.send_progress(self.client_id, event)
        
        logger.info(f"进度更新: {stage} - {message} ({percentage:.1f}%) [剩余约 {remaining_seconds:.0f}s]")
        
    def update_stage(self, stage: str, message: str, percentage: Optional[float] = None):
        """
        更新阶段（简化方法）
        
        update_progress 的简化版本，仅更新阶段和进度百分比，不改变步骤号。
        
        Args:
            stage: 阶段标识符
            message: 进度消息描述
            percentage: 进度百分比（可选）
        """
        self.update_progress(stage, message, percentage=percentage)
        
    def error(self, stage: str, message: str, details: Optional[Dict[str, Any]] = None):
        """
        发送错误
        
        向客户端发送错误事件。
        
        Args:
            stage: 发生错误的阶段标识符
            message: 错误消息描述
            details: 额外的详细信息字典，可选
        """
        event = create_error_event(stage, message, details)
        progress_manager.send_progress(self.client_id, event)
        logger.error(f"错误: {stage} - {message}")
        
    def complete(self, result: Dict[str, Any]):
        """
        完成任务
        
        向客户端发送完成事件，并记录总处理时间。
        
        Args:
            result: 任务结果数据字典
        """
        event = create_complete_event(result)
        progress_manager.send_progress(self.client_id, event)
        
        processing_time = (datetime.now() - self.start_time).total_seconds()
        logger.info(f"完成: 总用时 {processing_time:.2f}秒")


# 全局函数，用于在知识图谱构建过程中推送进度
def progress_callback(stage: str, message: str, percentage: Optional[float] = None):
    """
    进度回调函数 - 供外部调用
    
    全局进度回调函数，用于在知识图谱构建过程中推送进度。
    会将进度广播给所有监听器。
    
    注意：此函数会广播给所有监听器，如果只想向特定客户端发送进度，
    请使用 ProgressTracker 或直接调用 progress_manager.send_progress()。
    
    Args:
        stage: 阶段标识符，如 'initialization', 'document_loading', 'knowledge_graph' 等
        message: 进度消息描述
        percentage: 进度百分比（可选，0-100）
    
    Example:
        # 在知识图谱构建过程中使用
        progress_callback('initialization', '正在初始化知识图谱管理器...', 0)
        progress_callback('document_loading', '正在加载文档...', 20)
        progress_callback('knowledge_graph', '知识图谱构建完成', 100)
    """
    # 广播进度给所有监听器
    event = create_progress_event(stage, message, percentage)
    progress_manager.broadcast_progress(event)
    
    logger.info(f"构建进度: {stage} - {message}" + (f" ({percentage:.1f}%)" if percentage else ""))


def consume_sse_queue(q: queue.Queue, check_future: Optional[Callable[[], bool]] = None, heartbeat_interval: float = 1.0):
    """
    通用 SSE 队列消费者生成器
    
    从消息队列中消费消息并生成 SSE 格式的字符串。
    支持心跳机制和任务状态检查，防止连接超时和死循环。
    
    主要功能：
    - 队列消费：从队列中获取消息并转换为 SSE 格式
    - 心跳机制：在队列空闲时发送心跳注释，防止网关/浏览器超时
    - 任务检查：通过回调函数检查后台任务是否已完成
    - 自动终止：收到完成或错误事件时自动终止
    
    Args:
        q: 消息队列（queue.Queue）
        check_future: 可选的回调函数，用于检查后台任务状态。
                     如果返回 True，则停止循环。
                     用于防止回调丢失导致的死循环。
        heartbeat_interval: 心跳间隔（秒），默认为 1.0 秒
    
    Yields:
        str: SSE 格式的字符串
    
    Example:
        >>> q = queue.Queue()
        >>> # 在另一个线程中向队列发送消息
        >>> q.put({'type': 'progress', 'message': 'Processing...'})
        >>> q.put({'type': 'complete', 'result': {}})
        >>> 
        >>> # 在 Flask 路由中使用
        >>> @app.route('/stream')
        >>> def stream():
        >>>     def check_done():
        >>>         return task.done()
        >>>     return Response(consume_sse_queue(q, check_done), mimetype='text/event-stream')
    """
    while True:
        try:
            # 阻塞等待消息，设置超时防止死锁
            # 这里的超时也是一种心跳机制，确保连接活跃
            data = q.get(timeout=heartbeat_interval)
            yield sse_event(data)
            
            # 检查是否完成或出错
            msg_type = data.get('type')
            if msg_type in ['complete', 'error']:
                break
                
        except queue.Empty:
            # 队列空闲，检查任务是否已完成（防止回调丢失导致的死循环）
            if check_future and check_future():
                break
            
            # 发送心跳注释，防止网关/浏览器超时
            yield ": heartbeat\n\n"
            continue
