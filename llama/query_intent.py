"""
查询意图枚举模块

定义知识图谱查询的各种意图类型，用于意图识别和查询路径决策。

支持的查询意图：
- TREATMENT: 治疗方案、防控措施
- MECHANISM: 发病机制、病理生理
- SYMPTOM: 症状、体征、表现
- DIAGNOSIS: 诊断方法、检查参数
- PREVENTION: 预防措施、保健方法
- COMPLICATION: 并发症、副作用
- RISK_FACTOR: 危险因素、诱因
- GENERAL: 综合查询

使用示例：
    from llama.query_intent import QueryIntent
    
    # 获取意图的中文描述
    print(QueryIntent.TREATMENT.value)  # 输出: 治疗防控
    
    # 比较意图
    if intent == QueryIntent.TREATMENT:
        print("用户想了解治疗方案")
    
    # 遍历所有意图
    for intent in QueryIntent:
        print(f"{intent.name}: {intent.value}")
"""

from enum import Enum


class QueryIntent(Enum):
    """
    查询意图枚举
    
    定义知识图谱查询的各种意图类型，用于意图识别和查询路径决策。
    
    枚举值说明：
    - TREATMENT: 治疗防控 - 治疗方案、防控措施
    - MECHANISM: 发病机制 - 发病机制、病理生理
    - SYMPTOM: 症状表现 - 症状、体征、表现
    - DIAGNOSIS: 诊断检查 - 诊断方法、检查参数
    - PREVENTION: 预防保健 - 预防措施、保健方法
    - COMPLICATION: 并发症 - 并发症、副作用
    - RISK_FACTOR: 风险因素 - 危险因素、诱因
    - GENERAL: 综合查询 - 综合查询
    
    Attributes:
        value (str): 意图的中文描述
    """
    TREATMENT = "治疗防控"
    MECHANISM = "发病机制"
    SYMPTOM = "症状表现"
    DIAGNOSIS = "诊断检查"
    PREVENTION = "预防保健"
    COMPLICATION = "并发症"
    RISK_FACTOR = "风险因素"
    GENERAL = "综合查询"
