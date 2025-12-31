"""
增强的实体类型提取器 - 完全依赖LLM语义分析，无任何限制
"""

import logging
import os
from typing import List, Tuple, Dict, Any, Optional
import json
import re
import queue
import threading
import resource
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import EntityNode and Relation from llama_index.core
from llama_index.core.graph_stores.types import EntityNode, Relation
from llama_index.core.schema import BaseNode
from llama_index.core.indices.property_graph import DynamicLLMPathExtractor

logger = logging.getLogger(__name__)

try:
    from utils import parse_llm_output_with_types, safe_json_parse
except (ImportError, ModuleNotFoundError):
    logger.warning("enhanced_utils.py not found. Using basic fallback.")
    def parse_llm_output_with_types(llm_output: str) -> List[Dict[str, str]]:
        # 本地简易回退
        import re
        results = []
        pattern = r'"head"\s*:\s*"(.*?)"\s*,\s*"head_type"\s*:\s*"(.*?)"\s*,\s*"relation"\s*:\s*"(.*?)"\s*,\s*"tail"\s*:\s*"(.*?)"\s*,\s*"tail_type"\s*:\s*"(.*?)"'
        matches = re.findall(pattern, llm_output, re.DOTALL)
        for h, ht, r, t, tt in matches:
             results.append({"head":h, "head_type":ht, "relation":r, "tail":t, "tail_type":tt})
        return results
    
    def safe_json_parse(json_str: str) -> List[Dict[str, Any]]:
        try:
            return json.loads(json_str)
        except:
            return []

class EnhancedEntityExtractor:
    """增强的实体提取器 - 完全信任LLM语义分析"""
    
    @classmethod
    def extract_enhanced_triplets(cls, llm_output: str) -> List[Dict[str, Any]]:
        """提取增强的三元组，完全信任LLM的语义分析结果"""
        enhanced_triplets = []
        
        # 添加调试日志以查看LLM原始输出
        logger.info(f"LLM原始输出 (长度: {len(llm_output)}): {llm_output[:500]}...")
        
        # 使用 enhanced_utils 中的 parse_llm_output_with_types 
        # 这个函数已经集成了 safe_json_parse 和带类型的正则回退
        parsed_dicts = parse_llm_output_with_types(llm_output)
        
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
    """增强的解析函数，完全信任LLM的语义分析结果"""
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
            # 清理名称
            head_name = str(head_name).strip()
            tail_name = str(tail_name).strip()
            relation_type = str(relation_type).strip()
            
            # 验证：跳过纯标点或空的实体/关系
            invalid_symbols = {",", ".", "。", "，", "、", " ", "\\", "/", ";", ":", "?", "!", "'", "\"", "(", ")", "[", "]", "{", "}", "-", "_", "+", "=", "*", "&", "^", "%", "$", "#", "@", "~", "`", "<", ">", "|"}
            
            def is_invalid(text):
                if not text: return True
                if text in invalid_symbols: return True
                return all(char in invalid_symbols for char in text)

            if is_invalid(head_name) or is_invalid(tail_name) or is_invalid(relation_type):
                logger.warning(f"跳过无效实体/关系: '{head_name}' - '{relation_type}' - '{tail_name}'")
                continue

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
    Multi-stage LLM Extractor:
    1. Parallel Entity Recognition
    2. Producer-Consumer Relation Extraction
    """
    def __init__(
        self,
        llm: Any,
        entity_prompt: str,
        relation_prompt: str,
        num_workers: int = 4,
        max_triplets_per_chunk: int = 10,
        graph_store: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            llm=llm,
            extract_prompt=entity_prompt, # Placeholder
            parse_fn=None, # We implement custom logic
            num_workers=num_workers,
            max_triplets_per_chunk=max_triplets_per_chunk,
            **kwargs,
        )
        # Bypass Pydantic validation for custom fields
        object.__setattr__(self, "entity_prompt", entity_prompt)
        object.__setattr__(self, "relation_prompt", relation_prompt)
        object.__setattr__(self, "real_num_workers", num_workers)
        object.__setattr__(self, "graph_store", graph_store)
        
        # Memory monitoring config
        object.__setattr__(self, "memory_threshold_mb", 100)
        object.__setattr__(self, "peak_memory_usage", 0)
        
        # File write lock for saving JSON output
        object.__setattr__(self, "_file_lock", threading.Lock())

    def _safe_llm_call(self, prompt: str, max_retries: int = 3) -> str:
        """Call LLM with retry mechanism"""
        import time
        last_error = None
        for attempt in range(max_retries):
            try:
                response = self.llm.complete(prompt)
                return response.text
            except Exception as e:
                last_error = e
                logger.warning(f"LLM call failed (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1 * (attempt + 1))
        
        raise last_error

    def _save_json_output(self, node: BaseNode, triplets: List[Tuple]) -> None:
        """
        Securely save LLM output to a JSON file with metadata.
        Format: "original_filename-json.txt" in "llm_outputs/{date}/"
        """
        import datetime
        import os
        
        try:
            # 1. Prepare data
            file_name = node.metadata.get('file_name', 'unknown_file')
            safe_filename = os.path.basename(file_name)
            
            # Remove extension for cleaner naming if possible
            if '.' in safe_filename:
                base_name = safe_filename.rsplit('.', 1)[0]
            else:
                base_name = safe_filename
                
            json_data = {
                "node_id": node.node_id,
                "file_name": file_name,
                "timestamp": datetime.datetime.now().isoformat(),
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
            
            # 2. Prepare directory
            today_str = datetime.datetime.now().strftime("%Y-%m-%d")
            storage_dir = os.path.join(os.getcwd(), "llm_outputs", today_str)
            
            # Use lock for directory creation to avoid race conditions
            with self._file_lock:
                if not os.path.exists(storage_dir):
                    os.makedirs(storage_dir, exist_ok=True)
            
            # 3. Prepare filename
            output_filename = f"{base_name}-json.txt"
            output_path = os.path.join(storage_dir, output_filename)
            
            # 4. Format content
            current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            header = f"/* 处理时间: {current_time_str} */\n"
            content = json.dumps(json_data, ensure_ascii=False, indent=2)
            
            # 5. Write to file (with retry)
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    with self._file_lock:
                        with open(output_path, "a", encoding="utf-8") as f:
                            f.write(header)
                            f.write(content)
                            f.write("\n\n") # Separator
                    logger.info(f"✅ JSON output saved to: {output_path}")
                    break # Success
                except Exception as write_err:
                    if attempt < max_retries - 1:
                        time.sleep(0.1)
                    else:
                        raise write_err
                        
        except Exception as e:
            logger.error(f"Failed to save JSON output for node {node.node_id}: {e}")

    def extract(self, nodes: List[BaseNode]) -> List[Dict[str, Any]]:
        results = [{} for _ in range(len(nodes))]
        # Limit queue size for memory buffer control (approx 100 chunks)
        relation_queue = queue.Queue(maxsize=100)
        
        # Memory monitoring helper
        def check_memory():
            try:
                # Get memory usage in MB
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                if sys.platform == 'darwin':
                    usage_mb = rss / (1024 * 1024)
                else:
                    usage_mb = rss / 1024
                
                if usage_mb > self.peak_memory_usage:
                    object.__setattr__(self, "peak_memory_usage", usage_mb)
                    
                if usage_mb > self.memory_threshold_mb:
                    logger.warning(f"⚠️ Memory usage {usage_mb:.2f}MB exceeded threshold {self.memory_threshold_mb}MB")
            except Exception:
                pass

        # Stage 1: Entity Extraction (Producers)
        def entity_producer(node_idx, node):
            try:
                prompt = self.entity_prompt.format(text=node.text)
                output = self._safe_llm_call(prompt)
                entities = self._parse_entities(output)
                relation_queue.put((node_idx, node, entities))
                logger.debug(f"Stage 1 (Entity) done for node {node_idx}, found {len(entities)} entities")
            except Exception as e:
                logger.error(f"Stage 1 (Entity) failed for node {node_idx}: {e}")
                relation_queue.put((node_idx, node, []))

        # Stage 2: Relation Extraction (Consumers)
        def relation_consumer():
            while True:
                item = relation_queue.get()
                if item is None:
                    break
                node_idx, node, entities = item
                
                if not entities:
                    relation_queue.task_done()
                    continue
                    
                try:
                    entities_str = json.dumps(entities, ensure_ascii=False)
                    prompt = self.relation_prompt.format(text=node.text, entities=entities_str)
                    
                    output = self._safe_llm_call(prompt)
                    
                    # Use existing parsing logic
                    triplets = parse_llm_output_to_enhanced_triplets(output)
                    
                    # Save JSON output using the new robust method
                    self._save_json_output(node, triplets)

                    # If graph_store is available, write directly
                    if self.graph_store and triplets:
                        start_write = time.time()
                        try:
                            # Extract nodes and relations
                            head_nodes = [t[0] for t in triplets]
                            tail_nodes = [t[2] for t in triplets]
                            relations = [t[1] for t in triplets]
                            
                            # Deduplicate nodes by ID to reduce DB load
                            unique_nodes = {}
                            for n in head_nodes + tail_nodes:
                                unique_nodes[n.id] = n
                            
                            # Upsert to Neo4j
                            self.graph_store.upsert_nodes(list(unique_nodes.values()))
                            self.graph_store.upsert_relations(relations)
                            
                            write_time = time.time() - start_write
                            logger.info(f"✅ Directly stored {len(triplets)} triplets to Neo4j in {write_time:.2f}s")
                            
                            # Do NOT store in results to save memory
                            # Store empty dict or metadata if needed
                            # Return empty kg_triplets to satisfy PropertyGraphIndex contract
                            results[node_idx] = {
                                "kg_triplets": [], 
                                "saved_to_neo4j": True, 
                                "count": len(triplets)
                            }
                            
                        except Exception as db_err:
                            logger.error(f"❌ Failed to write to Neo4j: {db_err}. Falling back to memory.")
                            results[node_idx] = {"kg_triplets": triplets}
                    else:
                        # Fallback to memory storage
                        results[node_idx] = {"kg_triplets": triplets}
                    
                    logger.debug(f"Stage 2 (Relation) done for node {node_idx}, found {len(triplets)} triplets")
                    
                    # Check memory periodically
                    check_memory()
                    
                except Exception as e:
                    logger.error(f"Stage 2 (Relation) failed for node {node_idx}: {e}")
                finally:
                    relation_queue.task_done()

        # Start Consumers
        consumer_threads = []
        num_consumers = max(1, self.real_num_workers // 2)
        for _ in range(num_consumers):
            t = threading.Thread(target=relation_consumer)
            t.start()
            consumer_threads.append(t)
            
        # Start Producers
        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(iterable, **kwargs):
                return iterable

        with ThreadPoolExecutor(max_workers=self.real_num_workers) as executor:
            futures = [executor.submit(entity_producer, i, node) for i, node in enumerate(nodes)]
            for f in tqdm(as_completed(futures), total=len(nodes), desc="Entity Extraction", unit="node"):
                pass
        
        logger.info("Entity extraction (Stage 1) completed. Waiting for relation extraction (Stage 2)...")
        
        # Stop consumers
        for _ in range(num_consumers):
            relation_queue.put(None)
        
        for t in consumer_threads:
            t.join()
            
        return results

    def _parse_entities(self, output: str) -> List[Dict[str, str]]:
        try:
            return safe_json_parse(output)
        except:
            # Fallback regex
            import re
            matches = re.findall(r'\{\s*"name"\s*:\s*"(.*?)",\s*"type"\s*:\s*"(.*?)"\s*\}', output)
            return [{"name": m[0], "type": m[1]} for m in matches]
