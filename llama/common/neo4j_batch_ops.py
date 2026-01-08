"""
Neo4j 批处理操作工具，用于优化的数据库交互。

此模块提供以下工具：
- 批量节点和关系操作
- 优化的 Cypher 查询执行
- 事务管理
- 批量数据加载
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from llama_index.core.graph_stores import PropertyGraphStore
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

logger = logging.getLogger(__name__)


class Neo4jBatchOperations:
    """
    针对 Neo4j 图存储的优化批处理操作。
    
    通过批量操作和使用高效的 Cypher 查询来减少数据库往返次数。
    """
    
    def __init__(self, graph_store: PropertyGraphStore):
        """
        初始化批处理操作管理器。
        
        Args:
            graph_store: Neo4j 图存储实例
        """
        self.graph_store = graph_store
        self.batch_size = 1000  # 默认批处理大小
        self._verify_neo4j_store()
    
    def _verify_neo4j_store(self):
        """验证 graph_store 是否为 Neo4jPropertyGraphStore。"""
        if not isinstance(self.graph_store, Neo4jPropertyGraphStore):
            logger.warning("图存储不是 Neo4jPropertyGraphStore，批处理操作可能无法正常工作")
    
    def batch_upsert_nodes(self, 
                           nodes: List[Any],
                           batch_size: Optional[int] = None) -> int:
        """
        批量更新或插入节点以获得更好的性能。
        
        Args:
            nodes: 要更新或插入的节点列表
            batch_size: 批处理大小（如果为 None，则使用默认值）
            
        Returns:
            更新或插入的节点数量
            
        Examples:
            >>> batch_ops = Neo4jBatchOperations(graph_store)
            >>> batch_ops.batch_upsert_nodes(nodes, batch_size=500)
        """
        if not nodes:
            return 0
        
        batch_size = batch_size or self.batch_size
        total_upserted = 0
        
        # 分批处理
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i + batch_size]
            
            try:
                self.graph_store.upsert_nodes(batch)
                total_upserted += len(batch)
                logger.debug(f"Upserted batch {i//batch_size + 1}: {len(batch)} nodes")
            except Exception as e:
                logger.error(f"Failed to upsert batch {i//batch_size + 1}: {e}")
                # 继续处理下一批次
        
        logger.info(f"Batch upserted {total_upserted} nodes in {len(nodes)//batch_size + 1} batches")
        return total_upserted
    
    def batch_upsert_relations(self,
                             relations: List[Any],
                             batch_size: Optional[int] = None) -> int:
        """
        批量更新或插入关系以获得更好的性能。
        
        Args:
            relations: 要更新或插入的关系列表
            batch_size: 批处理大小（如果为 None，则使用默认值）
            
        Returns:
            更新或插入的关系数量
            
        Examples:
            >>> batch_ops = Neo4jBatchOperations(graph_store)
            >>> batch_ops.batch_upsert_relations(relations, batch_size=500)
        """
        if not relations:
            return 0
        
        batch_size = batch_size or self.batch_size
        total_upserted = 0
        
        # 分批处理
        for i in range(0, len(relations), batch_size):
            batch = relations[i:i + batch_size]
            
            try:
                self.graph_store.upsert_relations(batch)
                total_upserted += len(batch)
                logger.debug(f"Upserted batch {i//batch_size + 1}: {len(batch)} relations")
            except Exception as e:
                logger.error(f"Failed to upsert batch {i//batch_size + 1}: {e}")
                # 继续处理下一批次
        
        logger.info(f"Batch upserted {total_upserted} relations in {len(relations)//batch_size + 1} batches")
        return total_upserted
    
    def batch_upsert_triplets(self,
                              triplets: List[Tuple[Any, Any, Any]],
                              batch_size: Optional[int] = None) -> Dict[str, int]:
        """
        批量更新或插入三元组（节点和关系）。
        
        Args:
            triplets: (头节点, 关系, 尾节点) 三元组列表
            batch_size: 批处理大小（如果为 None，则使用默认值）
            
        Returns:
            包含更新或插入的节点和关系数量的字典
            
        Examples:
            >>> batch_ops = Neo4jBatchOperations(graph_store)
            >>> batch_ops.batch_upsert_triplets(triplets)
        """
        if not triplets:
            return {'nodes': 0, 'relations': 0}
        
        # 提取唯一的节点和关系
        head_nodes = [t[0] for t in triplets]
        tail_nodes = [t[2] for t in triplets]
        relations = [t[1] for t in triplets]
        
        # 对节点进行去重
        unique_nodes = {}
        for node in head_nodes + tail_nodes:
            if hasattr(node, 'id'):
                unique_nodes[node.id] = node
            else:
                unique_nodes[id(node)] = node
        
        # 更新或插入节点和关系
        node_count = self.batch_upsert_nodes(
            list(unique_nodes.values()),
            batch_size
        )
        relation_count = self.batch_upsert_relations(
            relations,
            batch_size
        )
        
        return {
            'nodes': node_count,
            'relations': relation_count,
            'triplets': len(triplets)
        }
    
    def execute_batch_cypher(self,
                              queries: List[str],
                              batch_size: Optional[int] = None) -> int:
        """
        批量执行多个 Cypher 查询。
        
        Args:
            queries: 要执行的 Cypher 查询列表
            batch_size: 批处理大小（如果为 None，则使用默认值）
            
        Returns:
            执行的查询数量
            
        Examples:
            >>> batch_ops = Neo4jBatchOperations(graph_store)
            >>> batch_ops.execute_batch_cypher(["MATCH (n) RETURN n", "MATCH (r) RETURN r"])
        """
        if not queries:
            return 0
        
        batch_size = batch_size or self.batch_size
        total_executed = 0
        
        # 分批执行
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            
            try:
                with self.graph_store._driver.session() as session:
                    for query in batch:
                        session.run(query)
                        total_executed += 1
                
                logger.debug(f"Executed batch {i//batch_size + 1}: {len(batch)} queries")
            except Exception as e:
                logger.error(f"Failed to execute batch {i//batch_size + 1}: {e}")
                # 继续处理下一批次
        
        logger.info(f"Batch executed {total_executed} queries in {len(queries)//batch_size + 1} batches")
        return total_executed
    
    def bulk_load_from_triplets(self,
                                triplets: List[Tuple[Any, Any, Any]],
                                use_unwind: bool = True) -> int:
        """
        使用 UNWIND 批量加载三元组以获得最佳性能。
        
        这是加载大量数据的最有效方式。
        
        Args:
            triplets: (头节点, 关系, 尾节点) 三元组列表
            use_unwind: 使用 UNWIND 进行批量加载（推荐）
            
        Returns:
            加载的三元组数量
            
        Examples:
            >>> batch_ops = Neo4jBatchOperations(graph_store)
            >>> batch_ops.bulk_load_from_triplets(triplets)
        """
        if not triplets:
            return 0
        
        if not use_unwind:
            # 回退到批量更新或插入
            result = self.batch_upsert_triplets(triplets)
            return result['triplets']
        
        try:
            with self.graph_store._driver.session() as session:
                # 准备 UNWIND 的数据
                data = []
                for head, relation, tail in triplets:
                    data.append({
                        'head_name': getattr(head, 'name', str(head)),
                        'head_id': getattr(head, 'id', str(id(head))),
                        'head_type': getattr(head, 'type', 'Entity'),
                        'relation_type': getattr(relation, 'label', 'RELATED_TO'),
                        'tail_name': getattr(tail, 'name', str(tail)),
                        'tail_id': getattr(tail, 'id', str(id(tail))),
                        'tail_type': getattr(tail, 'type', 'Entity')
                    })
                
                # 使用 UNWIND 进行批量加载
                query = """
                UNWIND $batch AS row
                MERGE (h:Entity {id: row.head_id})
                ON CREATE SET h.name = row.head_name, h.type = row.head_type
                MERGE (t:Entity {id: row.tail_id})
                ON CREATE SET t.name = row.tail_name, t.type = row.tail_type
                MERGE (h)-[r:RELATED_TO]->(t)
                ON CREATE SET r.label = row.relation_type
                """
                
                session.run(query, batch=data)
                logger.info(f"Bulk loaded {len(triplets)} triplets using UNWIND")
                return len(triplets)
                
        except Exception as e:
            logger.error(f"Bulk load failed: {e}")
            # 回退到批量更新或插入
            result = self.batch_upsert_triplets(triplets)
            return result['triplets']
    
    def delete_nodes_batch(self,
                          node_ids: List[str],
                          batch_size: Optional[int] = None) -> int:
        """
        批量删除节点。
        
        Args:
            node_ids: 要删除的节点 ID 列表
            batch_size: 批处理大小（如果为 None，则使用默认值）
            
        Returns:
            删除的节点数量
            
        Examples:
            >>> batch_ops = Neo4jBatchOperations(graph_store)
            >>> batch_ops.delete_nodes_batch(['id1', 'id2'])
        """
        if not node_ids:
            return 0
        
        batch_size = batch_size or self.batch_size
        total_deleted = 0
        
        for i in range(0, len(node_ids), batch_size):
            batch = node_ids[i:i + batch_size]
            
            try:
                self.graph_store.delete(ids=batch)
                total_deleted += len(batch)
                logger.debug(f"Deleted batch {i//batch_size + 1}: {len(batch)} nodes")
            except Exception as e:
                logger.error(f"Failed to delete batch {i//batch_size + 1}: {e}")
        
        logger.info(f"Batch deleted {total_deleted} nodes in {len(node_ids)//batch_size + 1} batches")
        return total_deleted
    
    def clear_database_batch(self) -> bool:
        """
        高效地清空整个数据库。
        
        Returns:
            成功返回 True，否则返回 False
            
        Examples:
            >>> batch_ops = Neo4jBatchOperations(graph_store)
            >>> batch_ops.clear_database_batch()
        """
        try:
            with self.graph_store._driver.session() as session:
                # 使用高效查询删除所有节点和关系
                session.run("MATCH (n) DETACH DELETE n")
                logger.info("Database cleared using batch operation")
                return True
        except Exception as e:
            logger.error(f"Failed to clear database: {e}")
            return False
    
    def get_node_count(self) -> int:
        """
        高效地获取节点总数。
        
        Returns:
            数据库中的节点数量
            
        Examples:
            >>> batch_ops = Neo4jBatchOperations(graph_store)
            >>> count = batch_ops.get_node_count()
        """
        try:
            with self.graph_store._driver.session() as session:
                result = session.run("MATCH (n) RETURN count(n) as count")
                record = result.single()
                return record['count'] if record else 0
        except Exception as e:
            logger.error(f"Failed to get node count: {e}")
            return 0
    
    def get_relation_count(self) -> int:
        """
        高效地获取关系总数。
        
        Returns:
            数据库中的关系数量
            
        Examples:
            >>> batch_ops = Neo4jBatchOperations(graph_store)
            >>> count = batch_ops.get_relation_count()
        """
        try:
            with self.graph_store._driver.session() as session:
                result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
                record = result.single()
                return record['count'] if record else 0
        except Exception as e:
            logger.error(f"Failed to get relation count: {e}")
            return 0
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        获取全面的数据库统计信息。
        
        Returns:
            包含数据库指标的字典
            
        Examples:
            >>> batch_ops = Neo4jBatchOperations(graph_store)
            >>> stats = batch_ops.get_database_stats()
        """
        stats = {
            'node_count': self.get_node_count(),
            'relation_count': self.get_relation_count()
        }
        
        try:
            with self.graph_store._driver.session() as session:
                # 获取节点标签分布
                result = session.run("MATCH (n) RETURN labels(n) as labels, count(n) as count")
                label_counts = {}
                for record in result:
                    labels = record['labels']
                    count = record['count']
                    label = labels[0] if labels else 'Unknown'
                    label_counts[label] = count
                
                stats['label_distribution'] = label_counts
                
                # 获取关系类型分布
                result = session.run("MATCH ()-[r]->() RETURN type(r) as type, count(r) as count")
                rel_counts = {}
                for record in result:
                    rel_type = record['type']
                    count = record['count']
                    rel_counts[rel_type] = count
                
                stats['relation_distribution'] = rel_counts
                
        except Exception as e:
            logger.error(f"Failed to get detailed stats: {e}")
        
        return stats


def create_batch_operations(graph_store: PropertyGraphStore,
                         batch_size: int = 1000) -> Neo4jBatchOperations:
    """
    创建批处理操作管理器的工厂函数。
    
    Args:
        graph_store: Neo4j 图存储实例
        batch_size: 操作的默认批处理大小
        
    Returns:
        Neo4jBatchOperations 实例
        
    Examples:
        >>> batch_ops = create_batch_operations(graph_store, batch_size=500)
    """
    return Neo4jBatchOperations(graph_store)
