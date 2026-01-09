#!/usr/bin/env python3
"""
测试JSON解析功能
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llama.common.json_utils import safe_json_parse, extract_json_from_text, fix_json_syntax, parse_llm_output


def test_json_parsing():
    """测试JSON解析功能"""
    
    # 测试1: 正常的JSON数组（三元组格式）
    test_json1 = '''[
  {
    "head": "近视",
    "head_type": "疾病",
    "relation": "表现为",
    "tail": "视物模糊",
    "tail_type": "症状体征"
  },
  {
    "head": "视力下降",
    "head_type": "症状体征",
    "relation": "表现为",
    "tail": "弱视",
    "tail_type": "疾病"
  }
]'''
    
    result1 = parse_llm_output(test_json1)
    print(f"测试1 - 正常JSON数组(三元组): {'✓' if result1 and len(result1) == 2 else '✗'}")
    if result1:
        print(f"  解析结果: {len(result1)} 个三元组")
        print(f"  第一个三元组: {result1[0]}")
    
    # 测试2: 带markdown代码块的JSON
    test_json2 = '''```json
[
  {
    "head": "视网膜",
    "head_type": "解剖结构",
    "relation": "表现为",
    "tail": "视物模糊",
    "tail_type": "症状体征"
  }
]
```'''
    
    result2 = parse_llm_output(test_json2)
    print(f"测试2 - Markdown代码块: {'✓' if result2 and len(result2) == 1 else '✗'}")
    if result2:
        print(f"  解析结果: {len(result2)} 个三元组")
    
    # 测试3: 带单引号的JSON（需要修复）
    test_json3 = """[{'head': 'test', 'head_type': 'type', 'relation': 'rel', 'tail': 'tail', 'tail_type': 'type'}]"""
    
    result3 = parse_llm_output(test_json3)
    print(f"测试3 - 单引号JSON: {'✓' if result3 and len(result3) == 1 else '✗'}")
    if result3:
        print(f"  解析结果: {len(result3)} 个三元组")
    
    # 测试4: 带尾随逗号的JSON（需要修复）
    test_json4 = """[{"head": "test", "head_type": "type", "relation": "rel", "tail": "tail", "tail_type": "type",}]"""
    
    result4 = parse_llm_output(test_json4)
    print(f"测试4 - 尾随逗号: {'✓' if result4 and len(result4) == 1 else '✗'}")
    if result4:
        print(f"  解析结果: {len(result4)} 个三元组")
    
    # 测试5: 混合文本中的JSON
    test_json5 = '''这是一些文本
[
  {
    "head": "屈光度",
    "head_type": "检查参数",
    "relation": "量化关系",
    "tail": "近视",
    "tail_type": "疾病"
  }
]
更多文本'''
    
    result5 = parse_llm_output(test_json5)
    print(f"测试5 - 混合文本: {'✓' if result5 and len(result5) == 1 else '✗'}")
    if result5:
        print(f"  解析结果: {len(result5)} 个三元组")
    
    # 测试6: 大型JSON（模拟LLM输出）
    test_json6 = '''[
  {
    "head": "近视",
    "head_type": "疾病",
    "relation": "表现为",
    "tail": "视物模糊",
    "tail_type": "症状体征"
  },
  {
    "head": "视力下降",
    "head_type": "症状体征",
    "relation": "表现为",
    "tail": "弱视",
    "tail_type": "疾病"
  },
  {
    "head": "眼轴长度",
    "head_type": "检查参数",
    "relation": "导致",
    "tail": "轴性近视",
    "tail_type": "疾病"
  },
  {
    "head": "屈光度",
    "head_type": "检查参数",
    "relation": "量化关系",
    "tail": "近视",
    "tail_type": "疾病"
  },
  {
    "head": "调节幅度",
    "head_type": "检查参数",
    "relation": "影响",
    "tail": "近视",
    "tail_type": "疾病"
  }
]'''
    
    result6 = parse_llm_output(test_json6)
    print(f"测试6 - 大型JSON (5个三元组): {'✓' if result6 and len(result6) == 5 else '✗'}")
    if result6:
        print(f"  解析结果: {len(result6)} 个三元组")
    
    # 测试7: 实体格式（非三元组）
    test_json7 = '''[
  {
    "name": "近视",
    "type": "疾病"
  },
  {
    "name": "视力下降",
    "type": "症状体征"
  }
]'''
    
    result7 = parse_llm_output(test_json7)
    print(f"测试7 - 实体格式: {'✓' if result7 and len(result7) == 2 else '✗'}")
    if result7:
        print(f"  解析结果: {len(result7)} 个实体")
        print(f"  第一个实体: {result7[0]}")
    
    print("\n所有测试完成！")


if __name__ == "__main__":
    test_json_parsing()
