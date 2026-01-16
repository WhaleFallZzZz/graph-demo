"""
错误处理和重试工具模块

提供统一的错误处理和重试机制功能。
"""

import logging
from typing import Optional, Callable, Any, Type, Tuple
from functools import wraps

logger = logging.getLogger(__name__)


def retry_on_failure(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None
) -> Callable:
    """
    失败重试装饰器
    
    在函数执行失败时自动重试，支持指数退避策略。
    
    Args:
        max_retries: 最大重试次数（不包括首次执行）
        delay: 初始延迟时间（秒）
        backoff_factor: 退避因子，每次重试的延迟时间 = delay * (backoff_factor ^ attempt)
        exceptions: 需要重试的异常类型元组
        on_retry: 重试时的回调函数，接收参数 (attempt, exception)
        
    Returns:
        装饰器函数
        
    Raises:
        最后一次尝试的异常（如果所有重试都失败）
        
    使用示例：
        ```python
        @retry_on_failure(max_retries=3, delay=1.0, backoff_factor=2.0)
        def fetch_data(url):
            # 可能失败的网络请求
            return requests.get(url)
        
        # 自定义重试回调
        def on_retry_callback(attempt, exception):
            print(f"第 {attempt} 次重试，错误: {exception}")
        
        @retry_on_failure(max_retries=5, on_retry=on_retry_callback)
        def process_data(data):
            # 可能失败的数据处理
            return complex_processing(data)
        ```
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            import time
            
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        if on_retry:
                            on_retry(attempt + 1, e)
                        
                        logger.warning(
                            f"{func.__name__} 执行失败，第 {attempt + 1} 次重试，"
                            f"延迟 {current_delay:.2f} 秒，错误: {e}"
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        logger.error(
                            f"{func.__name__} 执行失败，已达到最大重试次数 {max_retries}，"
                            f"最后错误: {e}"
                        )
            
            raise last_exception
        return wrapper
    return decorator
