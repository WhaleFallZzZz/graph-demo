"""
安全的 Neo4j 连接管理器
解决 Python 关闭时的析构函数异常问题
"""

import atexit
import sys
import threading
from typing import Optional, Dict, Any
from contextlib import contextmanager
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
import logging

logger = logging.getLogger(__name__)

class SafeNeo4jConnection:
    """线程安全的 Neo4j 连接管理器，带有正确的清理机制"""
    
    def __init__(self):
        self._connections: Dict[str, Neo4jPropertyGraphStore] = {}
        self._lock = threading.Lock()
        self._shutdown = False
        # 注册清理函数
        atexit.register(self._cleanup_all)
    
    def get_connection(self, **kwargs) -> Neo4jPropertyGraphStore:
        """获取或创建安全的 Neo4j 连接"""
        if self._shutdown:
            raise RuntimeError("Connection manager is shutting down")
            
        # 创建连接标识符
        conn_id = f"{kwargs.get('url', 'default')}_{kwargs.get('username', 'neo4j')}"
        
        with self._lock:
            if conn_id not in self._connections:
                # 创建新的安全包装连接
                self._connections[conn_id] = SafeNeo4jPropertyGraphStore(**kwargs)
                logger.info(f"Created new Neo4j connection: {conn_id}")
            
            return self._connections[conn_id]
    
    def _cleanup_all(self):
        """在 Python 关闭前清理所有连接"""
        self._shutdown = True
        with self._lock:
            for conn_id, connection in self._connections.items():
                try:
                    connection.safe_close()
                    logger.info(f"Safely closed connection: {conn_id}")
                except Exception as e:
                    logger.warning(f"Error closing connection {conn_id}: {e}")
            self._connections.clear()
    
    @contextmanager
    def connection_context(self, **kwargs):
        """上下文管理器，自动管理连接生命周期"""
        conn = None
        try:
            conn = self.get_connection(**kwargs)
            yield conn
        finally:
            if conn and hasattr(conn, 'safe_close'):
                conn.safe_close()


class SafeNeo4jPropertyGraphStore(Neo4jPropertyGraphStore):
    """安全包装的 Neo4jPropertyGraphStore，避免析构函数问题"""
    
    def __init__(self, **kwargs):
        # 调用父类构造函数
        super().__init__(**kwargs)
        self._closed = False
        self._close_lock = threading.Lock()
        
        # 保存连接参数，用于重新连接
        self._connection_params = kwargs
    
    def safe_close(self):
        """安全关闭连接，避免在 Python 关闭时出错"""
        with self._close_lock:
            if self._closed:
                return
                
            try:
                # 检查 Python 是否正在关闭
                if sys.meta_path is None:
                    logger.debug("Python is shutting down, skipping connection cleanup")
                    return
                
                # 安全地关闭连接
                if hasattr(self, '_driver') and self._driver:
                    self._driver.close()
                    logger.info("Neo4j driver safely closed")
                    
            except Exception as e:
                logger.warning(f"Error during connection cleanup: {e}")
            finally:
                self._closed = True
    
    def __del__(self):
        """重写析构函数，使用安全清理"""
        try:
            self.safe_close()
        except Exception:
            # 忽略所有异常，避免在 Python 关闭时出错
            pass
    
    def __enter__(self):
        """支持上下文管理器协议"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出时安全清理"""
        self.safe_close()


# 全局连接管理器实例
_connection_manager = SafeNeo4jConnection()

def get_safe_neo4j_connection(**kwargs) -> Neo4jPropertyGraphStore:
    """
    获取安全的 Neo4j 连接
    
    使用示例：
        # 方法1：直接获取连接
        graph_store = get_safe_neo4j_connection(
            username="neo4j",
            password="12345678", 
            url="bolt://localhost:7687"
        )
        
        # 方法2：使用上下文管理器
        with get_safe_neo4j_connection(...) as graph_store:
            # 使用 graph_store
            pass
            # 自动清理
    """
    return _connection_manager.get_connection(**kwargs)


@contextmanager
def neo4j_connection_context(**kwargs):
    """
    Neo4j 连接上下文管理器
    
    使用示例：
        with neo4j_connection_context(
            username="neo4j",
            password="12345678",
            url="bolt://localhost:7687"
        ) as graph_store:
            # 使用 graph_store
            pass
    """
    with _connection_manager.connection_context(**kwargs) as conn:
        yield conn