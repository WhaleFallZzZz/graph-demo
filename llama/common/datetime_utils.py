"""
统一的日期时间工具模块

提供统一的日期时间处理、格式化和计算功能
"""

from datetime import datetime, timedelta
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)


class DateTimeUtils:
    """统一日期时间工具类"""
    
    DATE_FORMAT = "%Y-%m-%d"
    TIME_FORMAT = "%H:%M:%S"
    DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
    ISO_FORMAT = "%Y-%m-%dT%H:%M:%S"
    
    @staticmethod
    def now() -> datetime:
        """获取当前时间"""
        return datetime.now()
    
    @staticmethod
    def now_str(format: str = DATETIME_FORMAT) -> str:
        """
        获取当前时间的字符串表示
        
        Args:
            format: 时间格式，默认为 "%Y-%m-%d %H:%M:%S"
            
        Returns:
            格式化的时间字符串
        """
        return datetime.now().strftime(format)
    
    @staticmethod
    def today_str() -> str:
        """获取今天的日期字符串"""
        return datetime.now().strftime(DateTimeUtils.DATE_FORMAT)
    
    @staticmethod
    def format_datetime(dt: datetime, format: str = DATETIME_FORMAT) -> str:
        """
        格式化日期时间
        
        Args:
            dt: 日期时间对象
            format: 时间格式
            
        Returns:
            格式化的时间字符串
        """
        return dt.strftime(format)
    
    @staticmethod
    def parse_datetime(dt_str: str, format: str = DATETIME_FORMAT) -> Optional[datetime]:
        """
        解析日期时间字符串
        
        Args:
            dt_str: 时间字符串
            format: 时间格式
            
        Returns:
            日期时间对象，解析失败返回None
        """
        try:
            return datetime.strptime(dt_str, format)
        except ValueError as e:
            logger.warning(f"Failed to parse datetime '{dt_str}' with format '{format}': {e}")
            return None
    
    @staticmethod
    def from_timestamp(timestamp: float) -> datetime:
        """
        从时间戳创建日期时间对象
        
        Args:
            timestamp: Unix时间戳
            
        Returns:
            日期时间对象
        """
        return datetime.fromtimestamp(timestamp)
    
    @staticmethod
    def to_timestamp(dt: datetime) -> float:
        """
        转换为时间戳
        
        Args:
            dt: 日期时间对象
            
        Returns:
            Unix时间戳
        """
        return dt.timestamp()
    
    @staticmethod
    def add_seconds(dt: datetime, seconds: float) -> datetime:
        """
        添加秒数
        
        Args:
            dt: 日期时间对象
            seconds: 秒数
            
        Returns:
            新的日期时间对象
        """
        return dt + timedelta(seconds=seconds)
    
    @staticmethod
    def add_minutes(dt: datetime, minutes: float) -> datetime:
        """
        添加分钟数
        
        Args:
            dt: 日期时间对象
            minutes: 分钟数
            
        Returns:
            新的日期时间对象
        """
        return dt + timedelta(minutes=minutes)
    
    @staticmethod
    def add_hours(dt: datetime, hours: float) -> datetime:
        """
        添加小时数
        
        Args:
            dt: 日期时间对象
            hours: 小时数
            
        Returns:
            新的日期时间对象
        """
        return dt + timedelta(hours=hours)
    
    @staticmethod
    def add_days(dt: datetime, days: float) -> datetime:
        """
        添加天数
        
        Args:
            dt: 日期时间对象
            days: 天数
            
        Returns:
            新的日期时间对象
        """
        return dt + timedelta(days=days)
    
    @staticmethod
    def elapsed_seconds(start: datetime, end: Optional[datetime] = None) -> float:
        """
        计算经过的秒数
        
        Args:
            start: 开始时间
            end: 结束时间，默认为当前时间
            
        Returns:
            经过的秒数
        """
        if end is None:
            end = datetime.now()
        return (end - start).total_seconds()
    
    @staticmethod
    def format_duration(seconds: float, show_ms: bool = False) -> str:
        """
        格式化持续时间
        
        Args:
            seconds: 秒数
            show_ms: 是否显示毫秒
            
        Returns:
            格式化的持续时间字符串
        """
        if seconds < 60:
            if show_ms:
                return f"{seconds:.3f}秒"
            return f"{seconds:.1f}秒"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}分钟"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}小时"
    
    @staticmethod
    def is_today(dt: datetime) -> bool:
        """
        检查是否是今天
        
        Args:
            dt: 日期时间对象
            
        Returns:
            是否是今天
        """
        return dt.date() == datetime.now().date()
    
    @staticmethod
    def is_yesterday(dt: datetime) -> bool:
        """
        检查是否是昨天
        
        Args:
            dt: 日期时间对象
            
        Returns:
            是否是昨天
        """
        return dt.date() == (datetime.now() - timedelta(days=1)).date()
    
    @staticmethod
    def days_between(start: datetime, end: datetime) -> int:
        """
        计算两个日期之间的天数
        
        Args:
            start: 开始日期
            end: 结束日期
            
        Returns:
            天数
        """
        return (end.date() - start.date()).days
    
    @staticmethod
    def get_start_of_day(dt: Optional[datetime] = None) -> datetime:
        """
        获取一天的开始时间（00:00:00）
        
        Args:
            dt: 日期时间对象，默认为当前时间
            
        Returns:
            一天的开始时间
        """
        if dt is None:
            dt = datetime.now()
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    
    @staticmethod
    def get_end_of_day(dt: Optional[datetime] = None) -> datetime:
        """
        获取一天的结束时间（23:59:59.999999）
        
        Args:
            dt: 日期时间对象，默认为当前时间
            
        Returns:
            一天的结束时间
        """
        if dt is None:
            dt = datetime.now()
        return dt.replace(hour=23, minute=59, second=59, microsecond=999999)


def get_current_timestamp() -> float:
    """获取当前时间戳"""
    return datetime.now().timestamp()


def format_now(format: str = DateTimeUtils.DATETIME_FORMAT) -> str:
    """格式化当前时间"""
    return DateTimeUtils.now_str(format)


def parse_iso_datetime(dt_str: str) -> Optional[datetime]:
    """解析ISO格式的日期时间字符串"""
    return DateTimeUtils.parse_datetime(dt_str, DateTimeUtils.ISO_FORMAT)


def format_iso_datetime(dt: datetime) -> str:
    """格式化为ISO格式的日期时间字符串"""
    return DateTimeUtils.format_datetime(dt, DateTimeUtils.ISO_FORMAT)
