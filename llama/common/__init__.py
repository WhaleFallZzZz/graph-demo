"""
知识图谱系统的通用工具包。

本包提供可复用的工具函数，包括：
- 文本处理和规范化
- JSON 解析和验证
- 文件操作和类型检测
- 并发处理和动态线程池
- 错误处理和重试机制
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
    validate_json_structure,
    json_to_csv,
    csv_to_json
)

from .file_utils import (
    get_file_hash
)

from .concurrent_utils import (
    DynamicThreadPool
)

from .error_handler import (
    retry_on_failure
)

__all__ = [
    'clean_text',
    'sanitize_for_neo4j',
    'safe_json_parse',
    'parse_llm_output',
    'fix_json_syntax',
    'extract_json_from_text',
    'validate_json_structure',
    'json_to_csv',
    'csv_to_json',
    'get_file_hash',
    'DynamicThreadPool',
    'retry_on_failure'
]
