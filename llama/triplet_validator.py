"""
三元组反向自检模块
使用轻量级模型 Qwen/Qwen2.5-7B-Instruct 进行三元组反向验证
策略：对于提取的 A -> B，检查文本中是否存在支持 B -> A 的证据
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from llama_index.core.graph_stores.types import EntityNode, Relation
from llama_index.core.schema import BaseNode

logger = logging.getLogger(__name__)


class TripletValidator:
    """三元组反向自检器"""
    
    def __init__(self, lightweight_llm: Any, documents: Optional[List[BaseNode]] = None):
        """
        初始化校验器
        
        Args:
            lightweight_llm: 轻量级LLM模型实例（用于校验）
            documents: 原始文档列表，用于查找支持证据
        """
        self.lightweight_llm = lightweight_llm
        self.documents = documents or []
        # 建立文档文本索引，方便快速查找
        self._doc_text_map = {doc.node_id: doc.text for doc in self.documents}
    
    def get_reverse_relation_prompt(self, head: str, relation: str, tail: str, text: str) -> str:
        """
        构建反向关系验证的prompt
        
        Args:
            head: 头实体
            relation: 原始关系
            tail: 尾实体
            text: 原文文本
            
        Returns:
            格式化的prompt
        """
        # 构建反向关系描述
        # 常见关系的反向关系映射
        reverse_relation_map = {
            "导致": "由...导致",
            "用于": "需要...用于",
            "包含": "属于",
            "表现为": "可见于",
            "检查依据": "可用于检查",
            "量化关系": "可通过...量化",
            "关联": "关联"
        }
        
        reverse_relation = reverse_relation_map.get(relation, f"{relation}的反向")
        
        prompt = f"""# 眼科视光知识图谱三元组反向自检任务

## 任务描述
你是眼科视光领域的知识图谱专家。请基于给定的医学文本，判断原始三元组是否具有逻辑自洽性。

## 原始三元组
- 头实体（Subject）: {head}
- 关系（Predicate）: {relation}
- 尾实体（Object）: {tail}
- 正向关系: {head} → {relation} → {tail}

## 反向关系验证目标
判断文本中是否存在支持反向关系的证据：
**"${tail} {reverse_relation} {head}"** （即：{tail} → {head}）

## 医学领域判断标准
1. **存在反向证据**（返回 true）：
   - 文本明确提到反向关系
   - 文本暗示可以通过 {tail} 反向推导或关联到 {head}
   - 医学逻辑上存在合理的反向关系（例如：症状可以指向疾病）

2. **不存在反向证据**（返回 false）：
   - 文本中没有任何信息支持反向关系
   - 关系本身是单向的且不符合医学常识
   - 例如："病理性近视 表现为 视网膜萎缩" 的反向 "视网膜萎缩 可见于 病理性近视" 可能成立
   - 但 "角膜塑形镜(OK镜) 用于 近视" 的反向 "近视 需要 角膜塑形镜" 不太合理（因为还有其他治疗方法）

3. **特殊情况**：
   - "表现为"关系的反向通常是合理的（症状可见于疾病）
   - "用于"关系的反向需要谨慎判断
   - "导致"关系的反向通常不成立（因果关系单向）

## 原文文本
{text}

## 输出格式
请仅返回 JSON 格式（不要包含任何其他文字）：
{{
  "has_reverse_evidence": true/false,
  "confidence": 0.0-1.0,
  "reason": "简要说明判断理由（医学角度）"
}}

注意：confidence 表示你对判断结果的置信度，0.5 表示不确定，1.0 表示非常确定。
"""
        return prompt
    
    def validate_triplet_reverse(self, head: str, relation: str, tail: str, text: str) -> Dict[str, Any]:
        """
        验证单个三元组的反向关系
        
        Args:
            head: 头实体
            relation: 关系
            tail: 尾实体
            text: 原文文本
            
        Returns:
            验证结果字典，包含 has_reverse_evidence, confidence, reason
        """
        try:
            prompt = self.get_reverse_relation_prompt(head, relation, tail, text)
            response = self.lightweight_llm.complete(prompt)
            response_text = response.text.strip()
            
            # 尝试解析JSON响应
            import json
            import re
            
            # 提取JSON部分（去除可能的markdown代码块）
            json_match = re.search(r'\{[^{}]*"has_reverse_evidence"[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                # 如果没有找到JSON，尝试直接解析整个响应
                json_str = response_text
            
            try:
                result = json.loads(json_str)
            except json.JSONDecodeError:
                # 如果JSON解析失败，尝试提取布尔值
                has_evidence = "true" in response_text.lower() or "存在" in response_text or "支持" in response_text
                result = {
                    "has_reverse_evidence": has_evidence,
                    "confidence": 0.5,
                    "reason": "无法解析JSON，基于关键词判断"
                }
            
            return {
                "has_reverse_evidence": result.get("has_reverse_evidence", False),
                "confidence": result.get("confidence", 0.5),
                "reason": result.get("reason", "未提供理由")
            }
            
        except Exception as e:
            logger.error(f"验证三元组反向关系失败 ({head} - {relation} - {tail}): {e}")
            # 出错时默认返回不通过，避免误判
            return {
                "has_reverse_evidence": False,
                "confidence": 0.0,
                "reason": f"验证过程出错: {str(e)}"
            }
    
    def find_source_text(self, head: str, tail: str, relation: str) -> Optional[str]:
        """
        查找包含该三元组的源文本
        
        Args:
            head: 头实体
            tail: 尾实体
            relation: 关系
            
        Returns:
            包含该三元组的文档文本，如果找不到则返回None
        """
        # 在文档中查找同时包含head和tail的文本块
        for doc in self.documents:
            text = doc.text
            if head in text and tail in text:
                return text
        
        # 如果找不到，返回第一个文档的文本（作为fallback）
        if self.documents:
            return self.documents[0].text
        
        return None
    
    def validate_triplets_batch(
        self, 
        triplets: List[Tuple[EntityNode, Relation, EntityNode]],
        sample_ratio: float = 0.3,
        core_entities: Optional[List[str]] = None,
        num_workers: int = 4
    ) -> List[Tuple[Tuple[EntityNode, Relation, EntityNode], Dict[str, Any]]]:
        """
        批量验证三元组（支持多worker并行处理）
        
        Args:
            triplets: 三元组列表
            sample_ratio: 抽样比例（0.0-1.0），用于随机抽取部分三元组进行验证
            core_entities: 核心实体列表，包含这些实体的三元组优先验证
            num_workers: 并行worker数量
            
        Returns:
            验证结果列表，每个元素为 (三元组, 验证结果)
        """
        if not triplets:
            return []
        
        # 选择需要验证的三元组
        candidates = []
        
        # 1. 优先选择包含核心实体的三元组
        if core_entities:
            for triplet in triplets:
                head, relation, tail = triplet
                if head.name in core_entities or tail.name in core_entities:
                    candidates.append(triplet)
        
        # 2. 如果候选数量不足，随机补充
        remaining_count = max(1, int(len(triplets) * sample_ratio)) - len(candidates)
        remaining_triplets = [t for t in triplets if t not in candidates]
        
        if remaining_count > 0 and remaining_triplets:
            random.shuffle(remaining_triplets)
            candidates.extend(remaining_triplets[:remaining_count])
        
        logger.info(f"反向自检: 从 {len(triplets)} 个三元组中选择了 {len(candidates)} 个进行验证（使用 {num_workers} 个worker并行处理）")
        
        # 使用线程池并行执行验证
        validation_results = []
        lock = threading.Lock()
        
        def validate_single_triplet(triplet: Tuple[EntityNode, Relation, EntityNode]) -> Tuple[Tuple[EntityNode, Relation, EntityNode], Dict[str, Any]]:
            """验证单个三元组"""
            head, relation, tail = triplet
            source_text = self.find_source_text(head.name, tail.name, relation.label)
            
            if not source_text:
                logger.warning(f"无法找到三元组的源文本: {head.name} - {relation.label} - {tail.name}")
                return (triplet, {
                    "has_reverse_evidence": False,
                    "confidence": 0.0,
                    "reason": "无法找到源文本"
                })
            
            # 执行反向验证
            result = self.validate_triplet_reverse(head.name, relation.label, tail.name, source_text)
            
            # 线程安全地记录日志（不记录"通过"，因为这是中间结果）
            with lock:
                logger.debug(
                    f"反向验证: {head.name} - {relation.label} - {tail.name} | "
                    f"反向证据: {result['has_reverse_evidence']} | "
                    f"置信度: {result['confidence']:.2f}"
                )
            
            return (triplet, result)
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_triplet = {executor.submit(validate_single_triplet, triplet): triplet for triplet in candidates}
            
            for future in as_completed(future_to_triplet):
                try:
                    result = future.result()
                    validation_results.append(result)
                except Exception as e:
                    triplet = future_to_triplet[future]
                    head, relation, tail = triplet
                    logger.error(f"验证三元组时发生错误 ({head.name} - {relation.label} - {tail.name}): {e}")
                    validation_results.append((triplet, {
                        "has_reverse_evidence": False,
                        "confidence": 0.0,
                        "reason": f"验证过程出错: {str(e)}"
                    }))
        
        logger.info(f"反向验证完成: 成功验证 {len(validation_results)} 个三元组")
        return validation_results
    
    def filter_invalid_triplets(
        self,
        triplets: List[Tuple[EntityNode, Relation, EntityNode]],
        validation_results: List[Tuple[Tuple[EntityNode, Relation, EntityNode], Dict[str, Any]]],
        confidence_threshold: float = 0.5
    ) -> Tuple[List[Tuple[EntityNode, Relation, EntityNode]], List[Tuple[EntityNode, Relation, EntityNode]]]:
        """
        根据验证结果过滤无效三元组
        
        Args:
            triplets: 原始三元组列表
            validation_results: 验证结果列表
            confidence_threshold: 置信度阈值，低于此值的视为无效
            
        Returns:
            (有效的三元组列表, 无效的三元组列表)
        """
        # 建立验证结果映射（使用可哈希的键）
        # 由于元组包含对象可能不可哈希，使用 (head.name, relation.label, tail.name) 作为键
        def get_triplet_key(triplet: Tuple[EntityNode, Relation, EntityNode]) -> Tuple[str, str, str]:
            """获取三元组的可哈希键"""
            head, relation, tail = triplet
            # 确保所有值都是字符串，处理可能的列表或None情况
            head_name = str(head.name) if head.name else ""
            if isinstance(head_name, list):
                head_name = str(head_name[0]) if head_name else ""
            relation_label = str(relation.label) if relation.label else ""
            if isinstance(relation_label, list):
                relation_label = str(relation_label[0]) if relation_label else ""
            tail_name = str(tail.name) if tail.name else ""
            if isinstance(tail_name, list):
                tail_name = str(tail_name[0]) if tail_name else ""
            return (head_name, relation_label, tail_name)
        
        validation_map = {}
        for triplet, result in validation_results:
            try:
                key = get_triplet_key(triplet)
                validation_map[key] = result
            except Exception as e:
                head, relation, tail = triplet
                logger.error(f"构建验证结果映射时出错: {head.name if hasattr(head, 'name') else head} - {relation.label if hasattr(relation, 'label') else relation} - {tail.name if hasattr(tail, 'name') else tail}: {e}")
                continue
        
        valid_triplets = []
        invalid_triplets = []
        
        for triplet in triplets:
            try:
                head, relation, tail = triplet
                key = get_triplet_key(triplet)
            except Exception as e:
                logger.error(f"处理三元组时出错: {e}, triplet type: {type(triplet)}")
                # 如果无法处理，默认保留
                valid_triplets.append(triplet)
                continue
            
            if key in validation_map:
                result = validation_map[key]
                # 判断逻辑：
                # 1. 如果有反向证据，保留（说明逻辑自洽）
                # 2. 如果没有反向证据，但置信度 >= 阈值，也保留（可能是单向关系，但模型有信心）
                # 3. 如果没有反向证据且置信度 < 阈值，过滤掉（低置信度的无效关系）
                has_reverse = result.get("has_reverse_evidence", False)
                confidence = result.get("confidence", 0.0)
                
                # 判断是否保留：有反向证据 OR 置信度 >= 阈值
                should_keep = has_reverse or confidence >= confidence_threshold
                
                if should_keep:
                    valid_triplets.append(triplet)
                    logger.debug(
                        f"✅ 保留三元组: {head.name} - {relation.label} - {tail.name} | "
                        f"反向证据: {has_reverse}, 置信度: {confidence:.2f} (阈值: {confidence_threshold:.2f})"
                    )
                else:
                    invalid_triplets.append(triplet)
                    logger.warning(
                        f"❌ 过滤无效三元组: {head.name} - {relation.label} - {tail.name} | "
                        f"反向证据: {has_reverse}, 置信度: {confidence:.2f} < 阈值: {confidence_threshold:.2f} | "
                        f"原因: {result.get('reason', '未知')}"
                    )
            else:
                # 未验证的三元组默认保留（因为我们只是抽样验证）
                valid_triplets.append(triplet)
        
        logger.info(f"反向自检过滤: 保留 {len(valid_triplets)} 个，过滤 {len(invalid_triplets)} 个三元组（阈值: {confidence_threshold:.2f}）")
        
        return valid_triplets, invalid_triplets
