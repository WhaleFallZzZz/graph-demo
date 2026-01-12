from enum import Enum

class QueryIntent(Enum):
    """查询意图枚举"""
    TREATMENT = "治疗防控"  # 治疗方案、防控措施
    MECHANISM = "发病机制"  # 发病机制、病理生理
    SYMPTOM = "症状表现"  # 症状、体征、表现
    DIAGNOSIS = "诊断检查"  # 诊断方法、检查参数
    PREVENTION = "预防保健"  # 预防措施、保健方法
    COMPLICATION = "并发症"  # 并发症、副作用
    RISK_FACTOR = "风险因素"  # 危险因素、诱因
    GENERAL = "综合查询"  # 综合查询
