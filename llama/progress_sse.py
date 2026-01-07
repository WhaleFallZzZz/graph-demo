#!/usr/bin/env python3
"""
SSE进度推送模块
提供知识图谱构建过程的实时进度推送功能
"""

import json
from typing import Dict, Any, Optional, Callable
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ProgressSSE:
    """SSE进度推送管理器"""
    
    def __init__(self):
        self.listeners = {}
        self.progress_cache = {}
        
    def add_listener(self, client_id: str, callback: Callable):
        """添加监听器"""
        self.listeners[client_id] = callback
        logger.info(f"添加监听器: {client_id}")
        
    def remove_listener(self, client_id: str):
        """移除监听器"""
        if client_id in self.listeners:
            del self.listeners[client_id]
            logger.info(f"移除监听器: {client_id}")
            
    def send_progress(self, client_id: str, data: Dict[str, Any]):
        """发送进度信息"""
        if client_id in self.listeners:
            try:
                # 缓存进度信息
                self.progress_cache[client_id] = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'data': data
                }
                
                # 调用回调函数发送SSE事件
                callback = self.listeners[client_id]
                callback(data)
                
            except Exception as e:
                logger.error(f"发送进度失败: {e}")
                
    def broadcast_progress(self, data: Dict[str, Any]):
        """广播进度信息给所有监听器"""
        for client_id in list(self.listeners.keys()):
            self.send_progress(client_id, data)
            
    def get_progress(self, client_id: str) -> Optional[Dict[str, Any]]:
        """获取指定客户端的进度信息"""
        return self.progress_cache.get(client_id)

# 全局进度管理器实例
progress_manager = ProgressSSE()

def sse_event(data: Dict[str, Any]) -> str:
    """生成SSE事件数据"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

def create_progress_event(
    stage: str, 
    message: str, 
    percentage: Optional[float] = None,
    details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """创建进度事件"""
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
    """创建错误事件"""
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
    """创建完成事件"""
    return {
        'type': 'complete',
        'stage': 'complete',
        'message': '处理完成',
        'result': result,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

class ProgressTracker:
    """进度跟踪器"""
    
    def __init__(self, client_id: str, total_steps: int = 8):
        self.client_id = client_id
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = datetime.now()
        self.stage_times = {}
        self.last_percentage = 0.0
        
    def update_progress(self, stage: str, message: str, step: Optional[int] = None, percentage: Optional[float] = None):
        """更新进度"""
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
        """更新阶段"""
        self.update_progress(stage, message, percentage=percentage)
        
    def error(self, stage: str, message: str, details: Optional[Dict[str, Any]] = None):
        """发送错误"""
        event = create_error_event(stage, message, details)
        progress_manager.send_progress(self.client_id, event)
        logger.error(f"错误: {stage} - {message}")
        
    def complete(self, result: Dict[str, Any]):
        """完成"""
        event = create_complete_event(result)
        progress_manager.send_progress(self.client_id, event)
        
        processing_time = (datetime.now() - self.start_time).total_seconds()
        logger.info(f"完成: 总用时 {processing_time:.2f}秒")

# 全局函数，用于在知识图谱构建过程中推送进度
def progress_callback(stage: str, message: str, percentage: Optional[float] = None):
    """进度回调函数 - 供外部调用"""
    # 广播进度给所有监听器
    event = create_progress_event(stage, message, percentage)
    progress_manager.broadcast_progress(event)
    
    logger.info(f"构建进度: {stage} - {message}" + (f" ({percentage:.1f}%)" if percentage else ""))