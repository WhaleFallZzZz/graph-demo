"""
统一的错误处理和日志工具模块

提供统一的错误处理、日志记录和异常管理功能
"""

import logging
import traceback
from typing import Optional, Callable, Any, Type, Tuple
from functools import wraps
from datetime import datetime

logger = logging.getLogger(__name__)


class ErrorHandler:
    """统一错误处理器"""
    
    @staticmethod
    def log_error(error: Exception, context: str = "", level: str = "error", include_traceback: bool = False) -> None:
        """
        记录错误日志
        
        Args:
            error: 异常对象
            context: 错误上下文信息
            level: 日志级别 (error, warning, info, debug)
            include_traceback: 是否包含堆栈跟踪
        """
        error_type = type(error).__name__
        error_msg = str(error)
        
        if context:
            log_msg = f"{context} - {error_type}: {error_msg}"
        else:
            log_msg = f"{error_type}: {error_msg}"
        
        log_func = getattr(logger, level.lower(), logger.error)
        
        if include_traceback:
            log_func(f"{log_msg}\n{traceback.format_exc()}")
        else:
            log_func(log_msg)
    
    @staticmethod
    def handle_exception(error: Exception, context: str = "", reraise: bool = False) -> Optional[str]:
        """
        处理异常并返回错误信息
        
        Args:
            error: 异常对象
            context: 错误上下文信息
            reraise: 是否重新抛出异常
            
        Returns:
            错误信息字符串
        """
        ErrorHandler.log_error(error, context)
        
        if reraise:
            raise error
        
        return f"{context}: {str(error)}" if context else str(error)


def safe_execute(
    default_return: Any = None,
    error_context: str = "",
    log_level: str = "error",
    reraise: bool = False,
    include_traceback: bool = False
) -> Callable:
    """
    安全执行装饰器，捕获并处理函数中的异常
    
    Args:
        default_return: 发生异常时的默认返回值
        error_context: 错误上下文信息
        log_level: 日志级别
        reraise: 是否重新抛出异常
        include_traceback: 是否包含堆栈跟踪
        
    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = f"{error_context} - {func.__name__}" if error_context else func.__name__
                ErrorHandler.log_error(e, context, log_level, include_traceback)
                
                if reraise:
                    raise
                
                return default_return
        return wrapper
    return decorator


def retry_on_failure(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None
) -> Callable:
    """
    失败重试装饰器
    
    Args:
        max_retries: 最大重试次数
        delay: 初始延迟时间（秒）
        backoff_factor: 退避因子
        exceptions: 需要重试的异常类型
        on_retry: 重试时的回调函数
        
    Returns:
        装饰器函数
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
                        
                        logger.warning(f"{func.__name__} 执行失败，第 {attempt + 1} 次重试，延迟 {current_delay:.2f} 秒")
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        logger.error(f"{func.__name__} 执行失败，已达到最大重试次数 {max_retries}")
            
            raise last_exception
        return wrapper
    return decorator


def retry_on_failure_with_strategy(
    max_retries: int = 3,
    retry_strategy: Optional[Callable[[Exception, int], float]] = None,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None
) -> Callable:
    """
    带自定义重试策略的失败重试装饰器
    
    Args:
        max_retries: 最大重试次数
        retry_strategy: 重试策略函数，接收异常和重试次数，返回等待时间（秒）
        exceptions: 需要重试的异常类型
        on_retry: 重试时的回调函数
        
    Returns:
        装饰器函数
    """
    def default_retry_strategy(error: Exception, attempt: int) -> float:
        """默认重试策略：根据错误类型采用不同的等待时间"""
        error_type = type(error).__name__
        error_str = str(error)
        
        if "RateLimitError" in error_type or "429" in error_str:
            return min(60, (2 ** attempt) * 5)
        elif "Timeout" in error_type or "timeout" in error_str.lower():
            return 5 * (attempt + 1)
        elif "ConnectionError" in error_type or "NetworkError" in error_type:
            return 10 * (attempt + 1)
        else:
            return 2 * (attempt + 1)
    
    strategy = retry_strategy or default_retry_strategy
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            import time
            
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        wait_time = strategy(e, attempt)
                        
                        if on_retry:
                            on_retry(attempt + 1, e)
                        
                        logger.warning(f"{func.__name__} 执行失败，第 {attempt + 1} 次重试，延迟 {wait_time:.2f} 秒")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"{func.__name__} 执行失败，已达到最大重试次数 {max_retries}")
            
            raise last_exception
        return wrapper
    return decorator


def log_execution_time(func: Callable) -> Callable:
    """
    记录函数执行时间的装饰器
    
    Args:
        func: 被装饰的函数
        
    Returns:
        装饰器函数
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = datetime.now()
        try:
            result = func(*args, **kwargs)
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"{func.__name__} 执行完成，耗时: {execution_time:.2f} 秒")
            return result
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            ErrorHandler.log_error(e, f"{func.__name__} 执行失败 (耗时: {execution_time:.2f} 秒)")
            raise
    return wrapper


class ErrorContext:
    """错误上下文管理器"""
    
    def __init__(self, context: str, log_level: str = "error", reraise: bool = False):
        """
        初始化错误上下文
        
        Args:
            context: 错误上下文信息
            log_level: 日志级别
            reraise: 是否重新抛出异常
        """
        self.context = context
        self.log_level = log_level
        self.reraise = reraise
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            ErrorHandler.log_error(exc_val, self.context, self.log_level)
            if self.reraise:
                return False
            return True
        return False


def validate_and_execute(
    validator: Callable[..., bool],
    error_message: str = "验证失败"
) -> Callable:
    """
    验证并执行装饰器
    
    Args:
        validator: 验证函数，返回 True 表示验证通过
        error_message: 验证失败时的错误信息
        
    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if not validator(*args, **kwargs):
                raise ValueError(error_message)
            return func(*args, **kwargs)
        return wrapper
    return decorator
