"""
测试Neo4j文本清理器
验证特殊字符处理的正确性
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'llama')))

from neo4j_text_sanitizer import Neo4jTextSanitizer, sanitize_for_neo4j


def test_sanitize_node_name():
    """测试节点名称清理"""
    print("\n" + "="*60)
    print("测试节点名称清理")
    print("="*60)
    
    test_cases = [
        # (输入, 预期包含的字符)
        ("'包含单引号'的实体", "'"),
        ('包含双引号"的实体', '"'),
        ("包含：冒号的实体", "："),
        ("包含*星号*的实体", "＊"),
        ("MATCH节点", "Entity_MATCH"),
        ("包含多个   空格的实体", "包含多个 空格的实体"),
        ("正常的实体名称", "正常的实体名称"),
        ("'模糊点'、'恢复点'、'破裂点'", "模糊点"),
        ("最低调节幅度 = 15-0.25*年龄", "最低调节幅度"),
    ]
    
    for original, expected_contains in test_cases:
        sanitized = Neo4jTextSanitizer.sanitize_node_name(original)
        
        # 验证不包含危险字符
        dangerous_chars = ["'", '"', "`", "\\"]
        has_dangerous = any(char in sanitized for char in dangerous_chars)
        
        status = "✅" if not has_dangerous and expected_contains in sanitized else "❌"
        print(f"{status} 原始: {original[:40]}")
        print(f"   清理后: {sanitized}")
        print()


def test_sanitize_relation_label():
    """测试关系标签清理"""
    print("\n" + "="*60)
    print("测试关系标签清理")
    print("="*60)
    
    test_cases = [
        ("用于", "用于"),
        ("包含：", "包含"),
        ("**检查依据**", "检查依据"),
        ("'适用于'", "适用于"),
        ("MERGE关系", "REL_MERGE"),
        ("定义为", "定义为"),
        ("是视网膜形成的重要组成部分", "是视网膜形成的重要组成部分"),
    ]
    
    for original, expected in test_cases:
        sanitized = Neo4jTextSanitizer.sanitize_relation_label(original)
        
        # 验证不包含危险字符
        dangerous_chars = ["'", '"', "*"]
        has_dangerous = any(char in sanitized for char in dangerous_chars)
        
        status = "✅" if not has_dangerous else "❌"
        print(f"{status} 原始: {original}")
        print(f"   清理后: {sanitized}")
        print(f"   预期: {expected}")
        print()


def test_sanitize_entity_type():
    """测试实体类型(Label)清理"""
    print("\n" + "="*60)
    print("测试实体类型(Label)清理")
    print("="*60)
    
    test_cases = [
        ("眼部疾病/异常", "眼部疾病_异常"),
        ("生理参数", "生理参数"),
        ("检查项目", "检查项目"),
        ("123数字开头", "Type_123数字开头"),
        ("包含特殊@字符#", "包含特殊_字符_"),
        ("WHERE", "Type_WHERE"),
    ]
    
    for original, expected in test_cases:
        sanitized = Neo4jTextSanitizer.sanitize_entity_type(original)
        
        # 验证只包含合法字符(字母、数字、下划线、汉字)
        import re
        is_valid = bool(re.match(r'^[\w\u4e00-\u9fff]+$', sanitized))
        
        status = "✅" if is_valid and expected == sanitized else "❌"
        print(f"{status} 原始: {original}")
        print(f"   清理后: {sanitized}")
        print(f"   预期: {expected}")
        print(f"   是否合法: {is_valid}")
        print()


def test_batch_sanitize():
    """测试批量清理"""
    print("\n" + "="*60)
    print("测试批量清理三元组")
    print("="*60)
    
    triplet = {
        "head": "'模糊点'、'恢复点'、'破裂点'",
        "head_type": "检查/参数",
        "relation": "**检查依据**",
        "tail": "Worth4点检测",
        "tail_type": "检查项目"
    }
    
    print("原始三元组:")
    for key, value in triplet.items():
        print(f"  {key}: {value}")
    
    sanitized = Neo4jTextSanitizer.batch_sanitize(triplet)
    
    print("\n清理后三元组:")
    for key, value in sanitized.items():
        print(f"  {key}: {value}")
    
    # 验证所有字段都被清理
    has_special = any(
        "'" in str(v) or '"' in str(v) or "*" in str(v) 
        for v in sanitized.values()
    )
    
    status = "✅" if not has_special else "❌"
    print(f"\n{status} 清理完成，无危险字符: {not has_special}")


def test_real_world_examples():
    """测试真实场景中的数据"""
    print("\n" + "="*60)
    print("测试真实场景数据")
    print("="*60)
    
    # 从您提供的截图中看到的实际数据
    real_examples = [
        {
            "head": "Worth4点检测",
            "head_type": "检查项目",
            "relation": "用于",
            "tail": "双眼同时视功能",
            "tail_type": "功能评估"
        },
        {
            "head": "调节滞后",
            "head_type": "眼部疾病/异常",
            "relation": "定义为",
            "tail": "晶体的调节反应小于调节刺激",
            "tail_type": "生理参数"
        },
        {
            "head": "远近距离水平融像",
            "head_type": "检查项目",
            "relation": "量化阈值",
            "tail": "'模糊点'、'恢复点'、'破裂点'",
            "tail_type": "测量指标"
        }
    ]
    
    for i, example in enumerate(real_examples, 1):
        print(f"\n示例 {i}:")
        print(f"  原始: {example['head']} -[{example['relation']}]-> {example['tail']}")
        
        sanitized = Neo4jTextSanitizer.batch_sanitize(example)
        
        print(f"  清理: {sanitized['head']} -[{sanitized['relation']}]-> {sanitized['tail']}")
        print(f"  类型: ({sanitized['head_type']}) -> ({sanitized['tail_type']})")


def test_cypher_injection_prevention():
    """测试防止Cypher注入"""
    print("\n" + "="*60)
    print("测试防止Cypher注入攻击")
    print("="*60)
    
    malicious_inputs = [
        "'; DROP DATABASE neo4j; --",
        "MATCH (n) DELETE n",
        "normal_entity'; CREATE (n:Hack); RETURN n; --",
    ]
    
    for malicious in malicious_inputs:
        sanitized = Neo4jTextSanitizer.sanitize_node_name(malicious)
        is_safe = Neo4jTextSanitizer.validate_text(sanitized)
        
        # 验证单引号和分号被清理
        has_dangerous = "'" in sanitized or ";" in sanitized
        
        status = "✅" if not has_dangerous else "❌"
        print(f"{status} 原始(恶意): {malicious}")
        print(f"   清理后: {sanitized}")
        print(f"   是否安全: {is_safe and not has_dangerous}")
        print()


if __name__ == "__main__":
    print("\n" + "🧪 Neo4j 文本清理器测试套件" + "\n")
    
    test_sanitize_node_name()
    test_sanitize_relation_label()
    test_sanitize_entity_type()
    test_batch_sanitize()
    test_real_world_examples()
    test_cypher_injection_prevention()
    
    print("\n" + "="*60)
    print("✅ 所有测试完成!")
    print("="*60)
