# AGENTS.md - Agent Guidelines for python_demo

This file provides guidelines for AI agents working on this repository.

## Build, Test, and Lint Commands

### Testing
- **Run all tests**: `python -m unittest discover tests/`
- **Run specific test file**: `python -m unittest tests.test_common_utils`
- **Run single test**: `python -m unittest tests.test_common_utils.TestTextUtils.test_clean_text`
- **Run tests in specific class**: `python -m unittest tests.test_common_utils.TestTextUtils`
- **Verbose output**: Add `-v` flag (e.g., `python -m unittest -v tests/`)

### Running the Server
- **Development mode**: `python -m llama.server`
- **Production (Gunicorn)**: `bash start_server.sh` (uses gunicorn with threaded mode)
- **Manual Gunicorn**: `gunicorn --worker-class gthread --workers 1 --threads 8 --bind 0.0.0.0:8001 --timeout 0 llama.server:app`
- **Windows (Waitress)**: `waitress-serve --port=8001 --call llama.server:app`

### Dependencies
- Install dependencies: `pip install -r requirements.txt`
- Llama-specific deps: `pip install -r llama/requirements.txt`

## Code Style Guidelines

### Imports
- Order: standard library → third-party → local imports
- Use explicit imports, avoid wildcard imports (`from module import *`)
- Common pattern: `from llama.common import clean_text, sanitize_for_neo4j`
- Each import on its own line for readability

### Type Hints
- Use type hints extensively for all function parameters and return values
- Common imports: `from typing import List, Dict, Any, Optional, Tuple, Callable, Union`
- Example: `def clean_text(text: str, remove_special: bool = True) -> str:`
- Use `Optional[T]` for nullable types

### Naming Conventions
- **Classes**: PascalCase (e.g., `EnhancedEntityExtractor`, `DynamicThreadPool`)
- **Functions/Variables**: snake_case (e.g., `clean_text`, `sanitize_for_neo4j`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `NEO4J_SPECIAL_CHARS`, `CLEAN_CHARS`)
- **Private methods**: Prefix with `_` (e.g., `_initialize_components`)
- **Modules**: lowercase_with_underscores (e.g., `text_utils.py`, `json_utils.py`)

### Docstrings
- Use triple-quoted docstrings (Chinese is common in this codebase)
- Include sections: Args, Returns, Raises, Examples/使用示例, Note
- Use detailed explanations for complex functions
- Example format:
```python
def sanitize_for_neo4j(text: str, max_length: int = 1000) -> str:
    """清理文本以便在 Neo4j 查询中安全使用。
    
    Args:
        text: 要清理的输入文本
        max_length: 允许的最大长度，超过此长度将被截断（默认：1000）
        
    Returns:
        对 Neo4j 安全的清理后文本，特殊字符已转义
    """
```

### Error Handling
- Use try/except blocks with specific exception types
- Always log errors using the logger module: `logger.error(f"Message: {e}")`
- Use custom exceptions when needed (e.g., `JSONParseError`)
- Use the `@retry_on_failure` decorator for transient failures
- Never expose secrets in error messages or logs

### Logging
- Initialize logger at module level: `logger = logging.getLogger(__name__)`
- Use appropriate log levels: DEBUG, INFO, WARNING, ERROR
- Configured via `setup_logging()` in config.py
- Logs are written to dated files in the `logs/` directory

### Code Organization
- Common utilities are in `llama/common/` module
- Import from common module: `from llama.common import clean_text, sanitize_for_neo4j`
- Use `__all__` in `__init__.py` to expose public API
- Factory pattern in `factories.py` for component creation

### Configuration
- All config is centralized in `llama/config.py`
- Use environment variables for secrets and deployment settings
- Load config: `from llama.config import API_CONFIG, NEO4J_CONFIG`
- Never commit `.env` files or actual secrets

### Key Patterns
- Use `DynamicThreadPool` from `llama.common` for thread management
- Use `retry_on_failure` decorator for unreliable operations
- Use `safe_json_parse` for LLM output parsing
- Use `sanitize_for_neo4j` for all text going to Neo4j
- Use `clean_text` for text normalization
- Use progress SSE streams for long-running operations

### File Type Support
- Supported formats: .txt, .docx, .pdf, .html, .md, .py, .json, .xml, .yaml
- Use `detect_file_type()` from `llama.file_type_detector` for validation

### Testing Patterns
- Use `unittest.TestCase` for test classes
- Test files in `tests/` directory
- Name test files: `test_<module_name>.py`
- Name test methods: `test_<function_name>`
- Use `setUp()` and `tearDown()` for test fixtures

### Security
- Never log or print API keys, passwords, or other secrets
- Validate all user input before processing
- Use parameterized queries for Neo4j to prevent injection
- Sanitize all text before database operations
- Check file types and sizes before processing uploads
