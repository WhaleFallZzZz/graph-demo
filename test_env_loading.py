#!/usr/bin/env python3
"""
测试环境变量加载
验证 .env 文件中的配置是否正确加载
"""

import sys
import os
from pathlib import Path

# 添加 llama 目录到路径
llama_dir = Path(__file__).parent / "llama"
sys.path.insert(0, str(llama_dir))

print("="*70)
print("测试环境变量加载")
print("="*70)
print()

# 测试 1: 直接使用 os.getenv
print("测试 1: 直接使用 os.getenv（未加载 .env）")
print(f"  VALIDATOR_ENABLE (直接): {os.getenv('VALIDATOR_ENABLE')}")
print()

# 测试 2: 加载 .env 后使用 os.getenv
from dotenv import load_dotenv
print("测试 2: 加载 .env 后使用 os.getenv")
load_dotenv()
print(f"  VALIDATOR_ENABLE (加载后): {os.getenv('VALIDATOR_ENABLE')}")
print()

# 测试 3: 从 config.py 导入 VALIDATOR_CONFIG
print("测试 3: 从 config.py 导入 VALIDATOR_CONFIG")
from config import VALIDATOR_CONFIG
print(f"  VALIDATOR_CONFIG['enable']: {VALIDATOR_CONFIG['enable']}")
print(f"  VALIDATOR_CONFIG['sample_ratio']: {VALIDATOR_CONFIG['sample_ratio']}")
print(f"  VALIDATOR_CONFIG['confidence_threshold']: {VALIDATOR_CONFIG['confidence_threshold']}")
print()

# 测试 4: 验证布尔值转换
print("测试 4: 验证布尔值转换")
test_values = ["true", "false", "True", "False", "TRUE", "FALSE", "yes", "no", "1", "0"]
for val in test_values:
    result = val.lower() == "true"
    print(f"  '{val}'.lower() == 'true' -> {result}")
print()

# 测试 5: 验证 .env 文件中的值
print("测试 5: 验证 .env 文件中的值")
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    print(f"  .env 文件路径: {env_file}")
    with open(env_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line.startswith("VALIDATOR_"):
                print(f"  行 {line_num}: {line}")
else:
    print(f"  ⚠️ .env 文件不存在: {env_file}")
print()

print("="*70)
print("结论")
print("="*70)
print()

if VALIDATOR_CONFIG['enable']:
    print("❌ VALIDATOR_CONFIG['enable'] = True")
    print("   反向自检将被启用")
else:
    print("✅ VALIDATOR_CONFIG['enable'] = False")
    print("   反向自检将被禁用")
