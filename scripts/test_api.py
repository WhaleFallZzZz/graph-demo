#!/usr/bin/env python3
"""测试 SiliconFlow API 连接"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
PROJECT_ROOT = Path(__file__).parent.parent
env_path = PROJECT_ROOT / '.env'
load_dotenv(env_path)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from llama.config import API_CONFIG

def test_api_connection():
    """测试 SiliconFlow API 连接"""
    from openai import OpenAI
    
    api_key = API_CONFIG["siliconflow"]["api_key"]
    base_url = "https://api.siliconflow.cn/v1"
    
    print(f"API Key: {api_key[:10]}..." if api_key else "API Key: None")
    print(f"Base URL: {base_url}")
    
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        print("\n正在测试 API 连接...")
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
            messages=[{"role": "user", "content": "你好"}],
            max_tokens=10
        )
        
        print("✓ API 连接成功!")
        print(f"响应: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"✗ API 连接失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_api_connection()
