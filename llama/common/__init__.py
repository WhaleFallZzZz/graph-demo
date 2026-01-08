"""
知识图谱系统的通用工具包。

本包提供可复用的工具函数，包括：
- 文本处理和规范化
- JSON 解析和验证
- 文件操作和类型检测
- 缓存管理
- 并发处理和动态线程池
- 日志和错误处理
"""

from .text_utils import (
    clean_text,
    sanitize_for_neo4j
)

from .json_utils import (
    safe_json_parse,
    parse_llm_output,
    fix_json_syntax,
    extract_json_from_text,
    validate_json_structure
)

from .file_utils import (
    get_file_hash
)

from .cache_utils import (
    CacheManager
)

from .concurrent_utils import (
    DynamicThreadPool,
    TaskManager
)

from .error_handler import (
    ErrorHandler,
    safe_execute,
    retry_on_failure,
    retry_on_failure_with_strategy,
    log_execution_time,
    ErrorContext,
    validate_and_execute
)

from .config_manager import (
    ConfigManager,
    NestedConfigManager,
    validate_config,
    merge_configs
)

from .datetime_utils import (
    DateTimeUtils,
    get_current_timestamp,
    format_now,
    parse_iso_datetime,
    format_iso_datetime
)

__all__ = [
    'clean_text',
    'sanitize_for_neo4j',
    'safe_json_parse',
    'parse_llm_output',
    'fix_json_syntax',
    'extract_json_from_text',
    'validate_json_structure',
    'get_file_hash',
    'CacheManager',
    'DynamicThreadPool',
    'TaskManager',
    'ErrorHandler',
    'safe_execute',
    'retry_on_failure',
    'retry_on_failure_with_strategy',
    'log_execution_time',
    'ErrorContext',
    'validate_and_execute',
    'ConfigManager',
    'NestedConfigManager',
    'validate_config',
    'merge_configs',
    'DateTimeUtils',
    'get_current_timestamp',
    'format_now',
    'parse_iso_datetime',
    'format_iso_datetime'
]
