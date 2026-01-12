"""
基于LLM的意图识别器
使用Few-Shot Learning进行智能意图分类
"""

import logging
from typing import Dict, List, Optional
from enum import Enum
import json

try:
    from .query_intent import QueryIntent
except ImportError:
    from query_intent import QueryIntent

logger = logging.getLogger(__name__)


class LLMIntentClassifier:
    """基于LLM的意图分类器"""
    
    def __init__(self, llm_instance=None):
        """
        初始化意图分类器
        
        Args:
            llm_instance: LLM实例，如果为None则使用默认LLM
        """
        self.llm = llm_instance
        self.intent_mapping = {
            "治疗防控": QueryIntent.TREATMENT,
            "发病机制": QueryIntent.MECHANISM,
            "症状表现": QueryIntent.SYMPTOM,
            "诊断检查": QueryIntent.DIAGNOSIS,
            "预防保健": QueryIntent.PREVENTION,
            "并发症": QueryIntent.COMPLICATION,
            "风险因素": QueryIntent.RISK_FACTOR,
            "综合查询": QueryIntent.GENERAL
        }
        
        # Few-Shot 示例
        self.few_shot_examples = [
            {
                "query": "阿托品对青少年近视的副作用",
                "intent": "并发症",
                "reasoning": "查询中明确提到了'副作用'，这是询问药物的不良反应，属于并发症类查询"
            },
            {
                "query": "阿托品的治疗方案",
                "intent": "治疗防控",
                "reasoning": "查询中提到了'治疗方案'，这是询问治疗方法和防控措施，属于治疗防控类查询"
            },
            {
                "query": "近视的发病机制",
                "intent": "发病机制",
                "reasoning": "查询中明确提到了'发病机制'，这是询问疾病的形成原因和原理，属于发病机制类查询"
            },
            {
                "query": "近视的症状表现",
                "intent": "症状表现",
                "reasoning": "查询中提到了'症状表现'，这是询问疾病的外在表现，属于症状表现类查询"
            },
            {
                "query": "如何预防近视",
                "intent": "预防保健",
                "reasoning": "查询中提到了'预防'，这是询问预防措施和保健方法，属于预防保健类查询"
            },
            {
                "query": "近视的并发症",
                "intent": "并发症",
                "reasoning": "查询中明确提到了'并发症'，这是询问疾病可能引发的其他疾病，属于并发症类查询"
            },
            {
                "query": "阿托品的不良反应",
                "intent": "并发症",
                "reasoning": "查询中提到了'不良反应'，这是询问药物的副作用，属于并发症类查询"
            },
            {
                "query": "近视的诊断检查",
                "intent": "诊断检查",
                "reasoning": "查询中提到了'诊断检查'，这是询问如何诊断和检查疾病，属于诊断检查类查询"
            },
            {
                "query": "近视的风险因素",
                "intent": "风险因素",
                "reasoning": "查询中提到了'风险因素'，这是询问导致疾病发生的危险因素，属于风险因素类查询"
            },
            {
                "query": "低浓度阿托品导致外斜",
                "intent": "并发症",
                "reasoning": "查询中描述了药物'导致'某种不良反应，这是描述副作用关系，属于并发症类查询"
            },
            {
                "query": "阿托品导致眼压升高",
                "intent": "并发症",
                "reasoning": "查询中描述了药物'导致'某种不良反应，这是描述副作用关系，属于并发症类查询"
            },
            {
                "query": "眼轴增长导致近视",
                "intent": "发病机制",
                "reasoning": "查询中描述了生理变化'导致'疾病，这是描述发病机制，属于发病机制类查询"
            },
            {
                "query": "近视表现为视力下降",
                "intent": "症状表现",
                "reasoning": "查询中描述了疾病'表现为'某种症状，这是描述症状表现，属于症状表现类查询"
            },
            {
                "query": "户外活动可以预防近视",
                "intent": "预防保健",
                "reasoning": "查询中提到了'预防'，这是描述预防措施，属于预防保健类查询"
            }
        ]
        
        # 意图描述
        self.intent_descriptions = {
            "治疗防控": "询问治疗方法、防控措施、矫正方案、改善手段等",
            "发病机制": "询问疾病形成的原因、原理、机制、病理过程等",
            "症状表现": "询问疾病的症状、表现、体征、外在现象等",
            "诊断检查": "询问诊断方法、检查项目、测量指标、筛查手段等",
            "预防保健": "询问预防措施、保健方法、注意事项、保护措施等",
            "并发症": "询问药物的副作用、不良反应、疾病并发症、风险危害等",
            "风险因素": "询问导致疾病的风险因素、诱因、危险因素、相关因素等",
            "综合查询": "综合性查询，涉及多个方面或意图不明确"
        }
    
    def _build_prompt(self, query: str) -> str:
        """
        构建Few-Shot提示词
        
        Args:
            query: 用户查询
            
        Returns:
            构建的提示词
        """
        # 构建Few-Shot示例
        examples_text = ""
        for i, example in enumerate(self.few_shot_examples[:8], 1):  # 使用前8个示例
            examples_text += f"""
示例 {i}:
查询: {example['query']}
意图: {example['intent']}
推理: {example['reasoning']}
"""
        
        prompt = f"""你是一个专业的医疗健康查询意图分类助手。请根据用户的查询内容，判断其意图类型。

## 意图类型说明：
{self._format_intent_descriptions()}

## Few-Shot 示例：
{examples_text}

## 当前查询：
查询: {query}

请按照以下格式输出你的分析：
```json
{{
    "intent": "意图类型",
    "confidence": 0.95,
    "reasoning": "推理过程"
}}
```

注意：
1. intent 必须是上述意图类型之一
2. confidence 是一个0到1之间的浮点数，表示分类的置信度
3. reasoning 是简短的推理过程说明
4. 对于涉及"副作用"、"不良反应"、"并发症"的查询，优先归类为"并发症"
5. 对于涉及"治疗"、"防控"、"方案"的查询，优先归类为"治疗防控"
"""
        return prompt
    
    def _format_intent_descriptions(self) -> str:
        """格式化意图描述"""
        descriptions = []
        for intent, desc in self.intent_descriptions.items():
            descriptions.append(f"- {intent}: {desc}")
        return "\n".join(descriptions)
    
    def classify(self, query: str) -> tuple[QueryIntent, float, str]:
        """
        使用LLM分类查询意图
        
        Args:
            query: 用户查询文本
            
        Returns:
            (意图, 置信度, 推理过程)
        """
        if self.llm is None:
            logger.warning("LLM实例未提供，使用默认分类逻辑")
            return self._fallback_classify(query)
        
        try:
            # 构建提示词
            prompt = self._build_prompt(query)
            
            # 调用LLM
            response = self.llm.complete(prompt)
            response_text = response.text.strip()
            
            logger.info(f"LLM意图分类原始输出: {response_text}")
            
            # 解析JSON响应
            result = self._parse_llm_response(response_text)
            
            if result:
                intent_name = result.get("intent", "综合查询")
                confidence = result.get("confidence", 0.5)
                reasoning = result.get("reasoning", "")
                
                # 映射到枚举
                intent = self.intent_mapping.get(intent_name, QueryIntent.GENERAL)
                
                logger.info(f"LLM意图分类: '{query}' -> {intent.value} (置信度: {confidence})")
                logger.debug(f"推理过程: {reasoning}")
                
                return intent, confidence, reasoning
            else:
                logger.warning("LLM响应解析失败，使用默认分类逻辑")
                return self._fallback_classify(query)
                
        except Exception as e:
            logger.error(f"LLM意图分类失败: {e}")
            return self._fallback_classify(query)
    
    def _parse_llm_response(self, response_text: str) -> Optional[Dict]:
        """
        解析LLM的JSON响应
        
        Args:
            response_text: LLM返回的文本
            
        Returns:
            解析后的字典，失败返回None
        """
        try:
            # 尝试提取JSON部分
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            else:
                json_text = response_text.strip()
            
            # 解析JSON
            result = json.loads(json_text)
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            return None
        except Exception as e:
            logger.error(f"响应解析失败: {e}")
            return None
    
    def _fallback_classify(self, query: str) -> tuple[QueryIntent, float, str]:
        """
        降级分类逻辑（基于关键词匹配）
        
        Args:
            query: 用户查询文本
            
        Returns:
            (意图, 置信度, 推理过程)
        """
        # 关键词映射
        intent_keywords = {
            QueryIntent.COMPLICATION: ["副作用", "不良反应", "并发症", "不良后果", "风险", "危害"],
            QueryIntent.TREATMENT: ["治疗", "防控", "控制", "矫正", "改善", "缓解", "方案", "方法"],
            QueryIntent.MECHANISM: ["机制", "原理", "原因", "导致", "引起", "形成", "发生", "发展"],
            QueryIntent.SYMPTOM: ["症状", "表现", "体征", "视物模糊", "视力下降", "疲劳"],
            QueryIntent.DIAGNOSIS: ["诊断", "检查", "检测", "测量", "参数", "指标", "筛查"],
            QueryIntent.PREVENTION: ["预防", "保健", "保护", "注意", "避免", "减少"],
            QueryIntent.RISK_FACTOR: ["风险因素", "诱因", "危险", "因素", "相关", "关联"]
        }
        
        # 计算每个意图的匹配分数
        intent_scores = {}
        for intent, keywords in intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query)
            if score > 0:
                intent_scores[intent] = score
        
        if not intent_scores:
            return QueryIntent.GENERAL, 0.3, "未匹配到明确意图，归类为综合查询"
        
        # 选择得分最高的意图
        max_intent = max(intent_scores.items(), key=lambda x: x[1])
        intent, score = max_intent
        
        # 计算置信度（归一化到0-1）
        confidence = min(0.5 + (score * 0.1), 0.9)
        
        reasoning = f"基于关键词匹配，找到 {score} 个相关关键词，归类为{intent.value}"
        
        logger.info(f"降级意图分类: '{query}' -> {intent.value} (置信度: {confidence})")
        
        return intent, confidence, reasoning