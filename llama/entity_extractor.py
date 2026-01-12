"""
增强的实体类型提取器 - 完全依赖LLM语义分析，无任何限制
"""

import logging
from typing import List, Tuple, Dict, Any, Optional
import json
import queue
import threading
import resource
import time
import sys
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Import EntityNode and Relation from llama_index.core
from llama_index.core.graph_stores.types import EntityNode, Relation
from llama_index.core.schema import BaseNode
from llama_index.core.indices.property_graph import DynamicLLMPathExtractor

# 导入 common 模块的工具
from llama.common import (
    safe_json_parse,
    parse_llm_output,
    clean_text,
    sanitize_for_neo4j,
    DynamicThreadPool,
    TaskManager,
    DateTimeUtils,
    retry_on_failure_with_strategy,
    retry_on_failure
)

try:
    from enhanced_entity_extractor import StandardTermMapper
except ImportError:
    # Fallback if file not found or circular import
    logger.warning("StandardTermMapper not found in enhanced_entity_extractor.py")
    class StandardTermMapper:
        @classmethod
        def process_triplets(cls, triplets):
            return triplets

logger = logging.getLogger(__name__)

class EnhancedEntityExtractor:
    """增强的实体提取器 - 完全信任LLM语义分析"""
    
    @classmethod
    def extract_enhanced_triplets(cls, llm_output: str) -> List[Dict[str, Any]]:
        """提取增强的三元组，完全信任LLM的语义分析结果"""
        enhanced_triplets = []
        
        # 添加调试日志以查看LLM原始输出
        logger.info(f"LLM原始输出 (长度: {len(llm_output)}): {llm_output[:500]}...")
        
        # 使用 common 模块中的 parse_llm_output
        parsed_dicts = parse_llm_output(llm_output)
        
        if parsed_dicts:
            for item in parsed_dicts:
                head = item.get("head", "").strip()
                head_type = item.get("head_type", "").strip()
                relation = item.get("relation", "").strip()
                tail = item.get("tail", "").strip()
                tail_type = item.get("tail_type", "").strip()
                
                # 只有当head, relation, tail都存在且不全是标点符号时才添加
                if head and relation and tail:
                    # 避免尾部是逗号等标点符号的无效提取
                    if tail in {",", ".", "。", "，", "、"}:
                         logger.warning(f"检测到无效的尾部实体(标点符号): '{tail}'，跳过该三元组")
                         continue

                    enhanced_triplets.append({
                        "head": head,
                        "head_type": head_type or "概念",
                        "relation": relation,
                        "tail": tail,
                        "tail_type": tail_type or "概念"
                    })
                    
                    logger.debug(f"提取LLM语义三元组: {head}({head_type}) - {relation} - {tail}({tail_type})")
        
        # 应用术语映射标准化 先注释
        enhanced_triplets = StandardTermMapper.process_triplets(enhanced_triplets)
        
        if not enhanced_triplets:
            logger.warning("未能从LLM输出中提取到任何有效的三元组")
             
        return enhanced_triplets
    
    @classmethod
    def validate_llm_entity_types(cls, enhanced_triplets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """验证LLM返回的实体类型 - 完全信任LLM，不再进行任何限制"""
        # 完全信任LLM的语义分析，不再验证类型是否在预定义列表中
        # 只进行基本的格式清理
        validated_triplets = []
        for triplet in enhanced_triplets:
            # 只进行基本的非空检查，完全信任LLM的语义判断
            head_type = triplet.get("head_type", "概念")
            tail_type = triplet.get("tail_type", "概念")
            
            # 只清理空白字符，不再进行任何类型限制
            triplet["head_type"] = head_type.strip() if head_type else "概念"
            triplet["tail_type"] = tail_type.strip() if tail_type else "概念"
            
            validated_triplets.append(triplet)
        
        return validated_triplets

# 修改 parse_llm_output_to_enhanced_triplets 函数以返回 EntityNode, Relation 对象
def parse_llm_output_to_enhanced_triplets(llm_output: str) -> List[Tuple[EntityNode, Relation, EntityNode]]:
    """增强的解析函数，完全信任LLM的语义分析结果，并清理特殊字符"""
    from neo4j_text_sanitizer import Neo4jTextSanitizer
    
    enhanced_triplets_dicts = EnhancedEntityExtractor.extract_enhanced_triplets(llm_output)
    
    # 验证LLM返回的实体类型 - 完全信任模式
    validated_triplets = EnhancedEntityExtractor.validate_llm_entity_types(enhanced_triplets_dicts)
    
    result_triplets = []
    for triplet_dict in validated_triplets:
        head_name = triplet_dict.get("head", "")
        head_type = triplet_dict.get("head_type", "概念")
        relation_type = triplet_dict.get("relation", "关联")
        tail_name = triplet_dict.get("tail", "")
        tail_type = triplet_dict.get("tail_type", "概念")
        
        if head_name and relation_type and tail_name:
            # 使用 common 模块中的 clean_text 进行基本清理
            head_name = clean_text(head_name, remove_special=False)
            tail_name = clean_text(tail_name, remove_special=False)
            relation_type = clean_text(relation_type, remove_special=False)

            # ---------------------------------------------------------
            # 强制映射层 (Standardization Error Fix) - 用户请求的强校验钩子
            # ---------------------------------------------------------
            try:
                # 再次尝试标准化，确保在创建节点前强制应用标准术语
                std_head = StandardTermMapper.standardize(head_name)
                if std_head in StandardTermMapper.STANDARD_ENTITIES:
                    if head_name != std_head:
                        logger.info(f"🔧 强制纠偏 (Head): {head_name} -> {std_head}")
                    head_name = std_head
                
                std_tail = StandardTermMapper.standardize(tail_name)
                if std_tail in StandardTermMapper.STANDARD_ENTITIES:
                    if tail_name != std_tail:
                        logger.info(f"🔧 强制纠偏 (Tail): {tail_name} -> {std_tail}")
                    tail_name = std_tail
            except Exception as e:
                logger.warning(f"StandardTermMapper 强校验失败: {e}")
            # ---------------------------------------------------------
            
            # 验证：跳过纯标点或空的实体/关系
            invalid_symbols = {",", ".", "。", "，", "、", " ", "\\", "/", ";", ":", "?", "!", "'", "\"", "(", ")", "[", "]", "{", "}", "-", "_", "+", "=", "*", "&", "^", "%", "$", "#", "@", "~", "`", "<", ">", "|"}
            
            def is_invalid(text):
                if not text: return True
                if text in invalid_symbols: return True
                return all(char in invalid_symbols for char in text)

            if is_invalid(head_name) or is_invalid(tail_name) or is_invalid(relation_type):
                logger.warning(f"跳过无效实体/关系: '{head_name}' - '{relation_type}' - '{tail_name}'")
                continue
            
            # 使用 common 模块中的 sanitize_for_neo4j 进行 Neo4j 安全清理
            # 记录清理前的值(用于日志对比)
            original_head = head_name
            original_tail = tail_name
            original_relation = relation_type
            original_head_type = head_type
            original_tail_type = tail_type
            
            # 清理节点名称
            head_name = Neo4jTextSanitizer.sanitize_node_name(head_name)
            tail_name = Neo4jTextSanitizer.sanitize_node_name(tail_name)
            
            # 清理关系标签（包含关系规范化）
            original_relation_length = len(relation_type)
            relation_type = Neo4jTextSanitizer.sanitize_relation_label(relation_type, max_length=10)
            
            # 清理实体类型(Label)
            head_type = Neo4jTextSanitizer.sanitize_entity_type(head_type)
            tail_type = Neo4jTextSanitizer.sanitize_entity_type(tail_type)
            
            # 如果清理后发生了变化，记录日志
            relation_changed = original_relation != relation_type
            relation_simplified = original_relation_length > 10 and len(relation_type) <= 10
            
            if (original_head != head_name or original_tail != tail_name or 
                relation_changed or original_head_type != head_type or 
                original_tail_type != tail_type):
                if relation_simplified:
                    logger.info(
                        f"🔧 关系简化: "
                        f"[{original_relation}] ({original_relation_length}字) -> [{relation_type}] ({len(relation_type)}字) | "
                        f"三元组: {head_name} - {tail_name}"
                    )
                else:
                    logger.debug(
                        f"🧹 字符清理: "
                        f"[{original_head}({original_head_type})] -> [{head_name}({head_type})], "
                        f"[{original_relation}] -> [{relation_type}], "
                        f"[{original_tail}({original_tail_type})] -> [{tail_name}({tail_type})]"
                    )
            
            # 再次验证清理后不为空
            if not head_name or not tail_name or not relation_type:
                logger.error(f"清理后实体/关系为空，跳过: {head_name} - {relation_type} - {tail_name}")
                continue
            # ===== 清理结束 =====

            logger.info(f"创建语义三元组: {head_name}({head_type}) - {relation_type} - {tail_name}({tail_type})")
                
            head_node = EntityNode(name=head_name, label=head_type)
            tail_node = EntityNode(name=tail_name, label=tail_type)
            
            relation = Relation(
                source_id=head_node.id,
                target_id=tail_node.id,
                label=relation_type
            )
            result_triplets.append((head_node, relation, tail_node))
        else:
            logger.warning(f"跳过无效三元组: {triplet_dict}")
            
    return result_triplets

# 保持原有的函数名兼容性
parse_dynamic_triplets = parse_llm_output_to_enhanced_triplets

class MultiStageLLMExtractor(DynamicLLMPathExtractor):
    """
    多阶段LLM提取器：
    1. 并行实体识别
    2. 生产者-消费者关系提取
    """
    def __init__(
        self,
        llm: Any,
        entity_prompt: str,
        relation_prompt: str,
        num_workers: int = 4,
        max_triplets_per_chunk: int = 15,
        graph_store: Optional[Any] = None,
        lightweight_llm: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            llm=llm,
            extract_prompt=entity_prompt, # 占位符
            parse_fn=None, # 我们实现自定义逻辑
            num_workers=num_workers,
            max_triplets_per_chunk=max_triplets_per_chunk,
            **kwargs,
        )
        # 绕过 Pydantic 验证以支持自定义字段
        object.__setattr__(self, "entity_prompt", entity_prompt)
        object.__setattr__(self, "relation_prompt", relation_prompt)
        object.__setattr__(self, "real_num_workers", num_workers)
        object.__setattr__(self, "graph_store", graph_store)
        object.__setattr__(self, "lightweight_llm", lightweight_llm or llm)
        
        # 内存监控配置
        object.__setattr__(self, "memory_threshold_mb", 100)
        object.__setattr__(self, "peak_memory_usage", 0)
        
        # 文件写入锁，用于保存JSON输出
        object.__setattr__(self, "_file_lock", threading.Lock())
        
        # 异步文件写入执行器
        object.__setattr__(self, "_write_executor", ThreadPoolExecutor(max_workers=2, thread_name_prefix="async_writer"))
        
        # Neo4j批量写入缓冲区
        object.__setattr__(self, "_node_buffer", {})
        object.__setattr__(self, "_relation_buffer", [])
        object.__setattr__(self, "_buffer_lock", threading.Lock())
        object.__setattr__(self, "_batch_write_threshold", 50)  # 每50个三元组批量写入一次

    @retry_on_failure_with_strategy(max_retries=3)
    def _call_llm_api(self, prompt: str, llm_instance: Any = None) -> str:
        """调用LLM API并返回结果"""
        target_llm = llm_instance or self.llm
        response = target_llm.complete(prompt)
        return response.text

    @retry_on_failure(max_retries=3, delay=0.1)
    def _write_to_file(self, output_path: str, header: str, content: str) -> None:
        """写入文件（带重试机制）"""
        with self._file_lock:
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(header)
                f.write(content)
                f.write("\n\n")

    def _write_to_file_async(self, output_path: str, header: str, content: str) -> None:
        """异步写入文件（优化版本）"""
        def write_task():
            try:
                with self._file_lock:
                    with open(output_path, "a", encoding="utf-8") as f:
                        f.write(header)
                        f.write(content)
                        f.write("\n\n")
                logger.debug(f"✅ 异步写入完成: {output_path}")
            except Exception as e:
                logger.error(f"❌ 异步写入失败: {e}")
                raise
        
        # 提交到线程池异步执行
        self._write_executor.submit(write_task)

    def _add_to_batch_buffer(self, nodes: List[EntityNode], relations: List[Relation]) -> bool:
        """添加节点关系到批量缓冲区，返回是否达到批量写入阈值"""
        with self._buffer_lock:
            # 添加节点到缓冲区（去重）
            for node in nodes:
                self._node_buffer[node.id] = node
            
            # 添加关系到缓冲区
            self._relation_buffer.extend(relations)
            
            # 检查是否达到批量写入阈值
            return len(self._relation_buffer) >= self._batch_write_threshold

    def _flush_batch_buffer(self) -> None:
        """将缓冲区的数据批量写入Neo4j"""
        with self._buffer_lock:
            if not self._node_buffer and not self._relation_buffer:
                return
            
            try:
                start_write = time.time()
                
                # 批量写入节点
                if self._node_buffer:
                    self.graph_store.upsert_nodes(list(self._node_buffer.values()))
                    logger.debug(f"✅ 批量写入 {len(self._node_buffer)} 个节点到 Neo4j")
                
                # 批量写入关系
                if self._relation_buffer:
                    self.graph_store.upsert_relations(self._relation_buffer)
                    logger.debug(f"✅ 批量写入 {len(self._relation_buffer)} 个关系到 Neo4j")
                
                write_time = time.time() - start_write
                logger.info(f"✅ 批量写入完成: {len(self._node_buffer)} 个节点, {len(self._relation_buffer)} 个关系, 耗时 {write_time:.2f}秒")
                
                # 清空缓冲区
                self._node_buffer.clear()
                self._relation_buffer.clear()
                
            except Exception as e:
                logger.error(f"❌ 批量写入 Neo4j 失败: {e}")
                # 清空缓冲区以避免重复写入
                self._node_buffer.clear()
                self._relation_buffer.clear()
                raise

    def _safe_llm_call(self, prompt: str, max_retries: int = 3, llm_instance: Any = None) -> str:
        """使用增强的重试机制和缓存调用LLM"""
        from llm_cache_manager import get_global_cache
        
        target_llm = llm_instance or self.llm
        
        # 尝试从缓存获取
        cache = get_global_cache()
        cached_result = cache.get(prompt, model_params={
            "temperature": 0.0,
            "model": getattr(target_llm, "model", "unknown")
        })
        
        if cached_result:
            logger.debug("使用缓存的LLM响应")
            return cached_result
        
        # 调用 LLM API（带重试）
        result = self._call_llm_api(prompt, llm_instance)
        
        # 缓存成功的结果
        cache.put(prompt, result, model_params={
            "temperature": 0.0,
            "model": getattr(target_llm, "model", "unknown")
        })
        
        return result

    def _save_json_output(self, node: BaseNode, triplets: List[Tuple]) -> None:
        """
        安全地将LLM输出保存到JSON文件，包含元数据。
        格式：在 "llm_outputs/{date}/" 目录下保存为 "original_filename-json.txt"
        """
        import os
        
        try:
            # 1. 准备数据
            file_name = node.metadata.get('file_name', 'unknown_file')
            safe_filename = os.path.basename(file_name)
            
            # 如果可能，移除扩展名以获得更清晰的命名
            if '.' in safe_filename:
                base_name = safe_filename.rsplit('.', 1)[0]
            else:
                base_name = safe_filename
                
            json_data = {
                "node_id": node.node_id,
                "file_name": file_name,
                "timestamp": DateTimeUtils.format_iso_datetime(DateTimeUtils.now()),
                "triplets": [
                    {
                        "head": t[0].name,
                        "head_type": t[0].label,
                        "relation": t[1].label,
                        "tail": t[2].name,
                        "tail_type": t[2].label
                    }
                    for t in triplets
                ]
            }
            
            # 2. 准备目录
            today_str = DateTimeUtils.today_str()
            storage_dir = os.path.join(os.getcwd(), "llm_outputs", today_str)
            
            # 使用锁创建目录以避免竞态条件
            with self._file_lock:
                if not os.path.exists(storage_dir):
                    os.makedirs(storage_dir, exist_ok=True)
            
            # 3. 准备文件名
            output_filename = f"{base_name}-json.txt"
            output_path = os.path.join(storage_dir, output_filename)
            
            # 4. 格式化内容
            current_time_str = DateTimeUtils.now_str()
            header = f"/* 处理时间: {current_time_str} */\n"
            content = json.dumps(json_data, ensure_ascii=False, indent=2)
            
            # 5. 写入文件（异步，带重试）
            try:
                self._write_to_file_async(output_path, header, content)
                logger.info(f"✅ JSON输出已保存（异步）到: {output_path}")
            except Exception as write_err:
                logger.error(f"无法将JSON输出写入文件: {write_err}")
                raise write_err
            
            # 监控日志
            process_time = time.time() - start_time
            logger.info(f"性能: 节点 {node.node_id[:8]} 处理耗时 {process_time:.4f}秒。提取了 {len(triplets)} 个三元组。")
                        
        except Exception as e:
            logger.error(f"无法为节点 {node.node_id} 保存JSON输出: {e}")

    def extract(self, nodes: List[BaseNode]) -> List[Dict[str, Any]]:
        results = [{} for _ in range(len(nodes))]
        # 限制队列大小以控制内存缓冲区（约100个文本块）
        relation_queue = queue.Queue(maxsize=100)
        
        # 内存监控辅助函数
        def check_memory():
            try:
                # 获取内存使用量（MB）
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                if sys.platform == 'darwin':
                    usage_mb = rss / (1024 * 1024)
                else:
                    usage_mb = rss / 1024
                
                if usage_mb > self.peak_memory_usage:
                    object.__setattr__(self, "peak_memory_usage", usage_mb)
                    
                if usage_mb > self.memory_threshold_mb:
                    logger.warning(f"⚠️ 内存使用量 {usage_mb:.2f}MB 超过阈值 {self.memory_threshold_mb}MB")
            except Exception:
                pass

        # 阶段1：批量实体提取（优化版）
        def batch_entity_producer():
            """批量实体提取 - 优化版本"""
            try:
                # 批量收集所有文本
                batch_size = 10
                for i in range(0, len(nodes), batch_size):
                    batch_nodes = nodes[i:i+batch_size]
                    batch_indices = list(range(i, min(i+batch_size, len(nodes))))
                    
                    # 构建批量prompt
                    batch_prompt = self._build_batch_entity_prompt(batch_nodes)
                    
                    # 使用轻量级LLM进行批量实体识别
                    output = self._safe_llm_call(batch_prompt, llm_instance=self.lightweight_llm)
                    
                    # 解析批量结果
                    batch_entities = self._parse_batch_entities(output, batch_indices)
                    
                    # 将结果放入队列
                    for node_idx, node in zip(batch_indices, batch_nodes):
                        entities = batch_entities.get(node_idx, [])
                        relation_queue.put((node_idx, node, entities))
                        logger.debug(f"阶段1（批量实体）完成节点 {node_idx}，发现 {len(entities)} 个实体")
                    
                    logger.info(f"✅ 批次 {i//batch_size + 1}: 处理了 {len(batch_nodes)} 个节点")
                    
            except Exception as e:
                logger.error(f"阶段1（批量实体）失败: {e}")
                # 回退到单独处理
                for node_idx, node in enumerate(nodes):
                    try:
                        prompt = self.entity_prompt.format(text=node.text)
                        output = self._safe_llm_call(prompt, llm_instance=self.lightweight_llm)
                        entities = self._parse_entities(output)
                        relation_queue.put((node_idx, node, entities))
                    except Exception as err:
                        logger.error(f"节点 {node_idx} 的回退实体提取失败: {err}")
                        relation_queue.put((node_idx, node, []))

        def _build_batch_entity_prompt(self, batch_nodes: List[BaseNode]) -> str:
            """构建批量实体提取的prompt"""
            prompt_parts = ["请从以下文本中提取实体，每个文本用编号标识：\n"]
            
            for idx, node in enumerate(batch_nodes):
                prompt_parts.append(f"[{idx}] {node.text}\n")
            
            prompt_parts.append("\n请以JSON格式返回结果，格式如下：\n")
            prompt_parts.append("{\n")
            prompt_parts.append('  "results": [\n')
            prompt_parts.append('    {"index": 0, "entities": [{"name": "实体名", "type": "实体类型"}]},\n')
            prompt_parts.append('    {"index": 1, "entities": [{"name": "实体名", "type": "实体类型"}]}\n')
            prompt_parts.append('  ]\n')
            prompt_parts.append("}\n")
            
            return "".join(prompt_parts)

        def _parse_batch_entities(self, output: str, batch_indices: List[int]) -> Dict[int, List[Dict[str, str]]]:
            """解析批量实体提取结果"""
            batch_entities = {}
            
            try:
                parsed = safe_json_parse(output)
                results = parsed.get("results", [])
                
                for result in results:
                    idx = result.get("index")
                    entities = result.get("entities", [])
                    if idx in batch_indices:
                        batch_entities[idx] = entities
                        
            except Exception as e:
                logger.error(f"解析批量实体失败: {e}")
                # 返回空字典，触发回退
                pass
            
            return batch_entities

        # 阶段2：批量关系提取（优化版）
        def relation_consumer():
            """批量关系提取 - 优化版本"""
            batch_buffer = []
            batch_size = 5
            batch_timeout = 2.0  # 秒
            
            while True:
                try:
                    # 从队列获取数据，带超时
                    item = relation_queue.get(timeout=batch_timeout)
                    
                    if item is None:
                        # 处理缓冲区中的剩余数据
                        if batch_buffer:
                            self._process_batch_relations(batch_buffer)
                            batch_buffer = []
                        break
                    
                    batch_buffer.append(item)
                    
                    # 达到批量大小时处理
                    if len(batch_buffer) >= batch_size:
                        self._process_batch_relations(batch_buffer)
                        batch_buffer = []
                        
                except queue.Empty:
                    # 超时后处理缓冲区中的数据
                    if batch_buffer:
                        self._process_batch_relations(batch_buffer)
                        batch_buffer = []
                except Exception as e:
                    logger.error(f"关系消费者错误: {e}")
                finally:
                    if item is not None:
                        relation_queue.task_done()

        def _process_batch_relations(self, batch_items: List[Tuple]) -> None:
            """批量处理关系提取"""
            if not batch_items:
                return
            
            logger.info(f"🔄 正在处理 {len(batch_items)} 个关系提取的批次")
            
            for node_idx, node, entities in batch_items:
                if not entities:
                    continue
                    
                try:
                    entities_str = json.dumps(entities, ensure_ascii=False)
                    prompt = self.relation_prompt.format(text=node.text, entities=entities_str)
                    
                    output = self._safe_llm_call(prompt)
                    
                    # 使用现有的解析逻辑
                    triplets = parse_llm_output_to_enhanced_triplets(output)
                    
                    # 使用新的稳健方法保存JSON输出
                    self._save_json_output(node, triplets)

                    # 如果 graph_store 可用，直接写入（使用批量缓冲区优化）
                    if self.graph_store and triplets:
                        # 提取节点和关系
                        head_nodes = [t[0] for t in triplets]
                        tail_nodes = [t[2] for t in triplets]
                        relations = [t[1] for t in triplets]
                        
                        # 添加到批量缓冲区
                        should_flush = self._add_to_batch_buffer(head_nodes + tail_nodes, relations)
                        
                        # 如果达到阈值，刷新缓冲区
                        if should_flush:
                            self._flush_batch_buffer()
                        
                        # 更新结果
                        results[node_idx] = {
                            "kg_triplets": [], 
                            "saved_to_neo4j": True, 
                            "count": len(triplets)
                        }
                    else:
                        # 回退到内存存储
                        results[node_idx] = {"kg_triplets": triplets}
                    
                    logger.debug(f"阶段2（关系）完成节点 {node_idx}，发现 {len(triplets)} 个三元组")
                    
                    # 定期检查内存
                    check_memory()
                    
                except Exception as e:
                    logger.error(f"节点 {node_idx} 的阶段2（关系）失败: {e}")

        # 启动消费者
        consumer_threads = []
        num_consumers = max(1, self.real_num_workers // 2)
        for _ in range(num_consumers):
            t = threading.Thread(target=relation_consumer)
            t.start()
            consumer_threads.append(t)
            
        # 启动批量生产者（优化版）
        logger.info("启动批量实体提取（阶段1）...")
        batch_entity_producer()
        logger.info("批量实体提取（阶段1）完成。等待关系提取（阶段2）...")
        
        # 停止消费者
        for _ in range(num_consumers):
            relation_queue.put(None)
        
        for t in consumer_threads:
            t.join()
        
        # 将剩余的批量缓冲区刷新到Neo4j
        if self.graph_store:
            logger.info("将剩余的批量缓冲区刷新到Neo4j...")
            self._flush_batch_buffer()
            
        return results

    def _parse_entities(self, output: str) -> List[Dict[str, str]]:
        try:
            return safe_json_parse(output)
        except:
            # 回退到正则表达式
            import re
            matches = re.findall(r'\{\s*"name"\s*:\s*"(.*?)",\s*"type"\s*:\s*"(.*?)"\s*\}', output)
            return [{"name": m[0], "type": m[1]} for m in matches]
