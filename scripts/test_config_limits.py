
import sys
import os
from pathlib import Path

# 设置工作目录
work_dir = Path("/Users/whalefall/Documents/workspace/python_demo")
sys.path.insert(0, str(work_dir))

try:
    from llama.config import get_rate_limit, MODEL_RATE_LIMITS
    print("✅ 成功导入 config 模块")
    
    models_to_test = [
        "BAAI/bge-m3",
        "BAAI/bge-reranker-v2-m3",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "deepseek-ai/DeepSeek-OCR",
        "unknown-model"
    ]
    
    for model in models_to_test:
        limit = get_rate_limit(model)
        print(f"\n模型: {model}")
        print(f"  RPM: {limit['rpm']}")
        print(f"  Delay: {limit['request_delay']:.4f}s")
        print(f"  Retries: {limit['max_retries']}")

except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
