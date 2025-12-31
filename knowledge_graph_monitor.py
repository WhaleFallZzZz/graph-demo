#!/usr/bin/env python3
"""
知识图谱文件监控自动化脚本
实现对指定目录的实时监控，自动触发知识图谱生成流程，并确保文件处理的幂等性
"""

import os
import sys
import json
import time
import hashlib
import logging
import argparse
import requests
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Set
from threading import Lock, Thread
from concurrent.futures import ThreadPoolExecutor
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

# 配置日志系统
def setup_logging(log_dir: str = None) -> logging.Logger:
    """设置日志配置，按日期生成日志文件
    
    Args:
        log_dir: 日志目录路径，默认为当前目录下的 logs 文件夹
    """
    if log_dir is None:
        log_dir = os.path.join(os.getcwd(), "logs")
    
    # 确保日志目录存在
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 生成带日期的日志文件名
    current_date = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(log_dir, f"monitor_{current_date}.log")
    
    # 配置日志格式
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


class FileProcessor:
    """文件处理器 - 负责文件上传和知识图谱构建"""
    
    def __init__(self, upload_api: str, build_api: str, max_retries: int = 3, retry_delay: float = 5.0):
        """
        初始化文件处理器
        
        Args:
            upload_api: 文件上传接口地址
            build_api: 知识图谱构建接口地址
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
        """
        self.upload_api = upload_api
        self.build_api = build_api
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = logging.getLogger(__name__)
        self.processing_lock = Lock()  # 防止同一文件被重复处理
        self.processing_files: Set[str] = set()  # 正在处理的文件集合
    
    def generate_file_hash(self, file_path: str) -> str:
        """
        生成文件的唯一哈希值（使用MD5）
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件的MD5哈希值（十六进制字符串）
        """
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            self.logger.error(f"生成文件哈希失败 {file_path}: {e}")
            raise
    
    def check_network_connection(self) -> bool:
        """
        检查网络连接是否可用
        
        Returns:
            True表示网络可用，False表示不可用
        """
        try:
            # 尝试连接上传接口的主机
            from urllib.parse import urlparse
            parsed = urlparse(self.upload_api)
            host = parsed.netloc.split(':')[0] if ':' in parsed.netloc else parsed.netloc
            
            # 简单的网络检查（ping或HTTP请求）
            response = requests.get(f"{parsed.scheme}://{host}", timeout=5)
            return True
        except Exception:
            try:
                # 如果直接连接失败，尝试连接通用网站
                requests.get("https://www.baidu.com", timeout=5)
                return True
            except Exception:
                return False
    
    def call_upload_api(self, file_path: str, file_hash: str) -> Optional[Dict[str, Any]]:
        """
        调用文件上传接口
        
        Args:
            file_path: 要上传的文件路径
            file_hash: 文件的哈希值
            
        Returns:
            上传接口的响应数据，失败返回None
        """
        if not self.check_network_connection():
            self.logger.error(f"网络连接不可用，无法上传文件: {file_path}")
            return None
        
        try:
            with open(file_path, 'rb') as f:
                files = {'file': (os.path.basename(file_path), f, 'application/octet-stream')}
                data = {'file_hash': file_hash}
                
                response = requests.post(
                    self.upload_api,
                    files=files,
                    data=data,
                    timeout=120  # 上传超时时间120秒
                )
                
                response.raise_for_status()
                result = response.json()
                
                if result.get('success') or 'file_url' in result.get('data', {}).get('file_info', {}):
                    self.logger.info(f"文件上传成功: {file_path}, 哈希: {file_hash}")
                    return result
                else:
                    self.logger.error(f"文件上传失败: {file_path}, 响应: {result}")
                    return None
                    
        except requests.exceptions.RequestException as e:
            self.logger.error(f"上传接口调用失败 {file_path}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"上传文件时发生异常 {file_path}: {e}")
            return None
    
    def call_build_api(self, file_url: str, file_hash: str) -> bool:
        """
        调用知识图谱构建接口
        
        Args:
            file_url: 文件在COS上的URL
            file_hash: 文件的哈希值
            
        Returns:
            True表示构建成功，False表示失败
        """
        if not self.check_network_connection():
            self.logger.error(f"网络连接不可用，无法构建知识图谱: {file_url}")
            return False
        
        try:
            data = {
                'file_url': file_url,
                'file_hash': file_hash
            }
            
            response = requests.post(
                self.build_api,
                json=data,
                timeout=6000,  # 构建超时时间6000秒（100分钟）
                stream=True  # 支持SSE流式响应
            )
            
            response.raise_for_status()
            
            # 处理SSE流式响应
            success = False
            for line in response.iter_lines(decode_unicode=True):
                if not line or line.strip() == '':
                    continue  # 跳过空行
                
                if line.startswith(':'):
                    continue  # 跳过注释行
                
                if line.startswith('data: '):
                    try:
                        event_data = json.loads(line[6:])  # 去掉 'data: ' 前缀
                        event_type = event_data.get('type', '')
                        
                        if event_type == 'complete':
                            success = True
                            self.logger.info(f"知识图谱构建完成: {file_url}")
                            break
                        elif event_type == 'error':
                            error_msg = event_data.get('message', '未知错误')
                            self.logger.error(f"知识图谱构建失败: {file_url}, 错误: {error_msg}")
                            break
                        elif event_type == 'progress':
                            progress = event_data.get('progress', 0)
                            stage = event_data.get('stage', '')
                            message = event_data.get('message', '')
                            self.logger.debug(f"构建进度: {stage} - {message} ({progress}%)")
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"解析SSE事件数据失败: {line[:100]}, 错误: {e}")
                        continue
                    except Exception as e:
                        self.logger.warning(f"处理SSE事件时出错: {e}")
                        continue
            
            # 如果没有收到complete或error事件，记录警告
            if not success:
                self.logger.warning(f"未收到明确的完成或错误事件: {file_url}")
            
            return success
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"构建接口调用失败 {file_url}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"构建知识图谱时发生异常 {file_url}: {e}")
            return False
    
    def process_file(self, file_path: str, mark_file_path: str) -> bool:
        """
        处理单个文件：上传并构建知识图谱
        
        Args:
            file_path: 要处理的文件路径
            mark_file_path: 标记文件路径
            
        Returns:
            True表示处理成功，False表示失败
        """
        file_key = os.path.abspath(file_path)
        
        # 检查文件是否正在处理
        with self.processing_lock:
            if file_key in self.processing_files:
                self.logger.warning(f"文件正在处理中，跳过: {file_path}")
                return False
            self.processing_files.add(file_key)
        
        try:
            # 生成文件哈希
            self.logger.info(f"开始处理文件: {file_path}")
            file_hash = self.generate_file_hash(file_path)
            self.logger.info(f"文件哈希值: {file_hash}")
            
            # 创建处理中标记文件
            self.create_processing_mark(mark_file_path, file_hash, "processing")
            
            # 重试机制：上传文件
            upload_result = None
            for attempt in range(1, self.max_retries + 1):
                upload_result = self.call_upload_api(file_path, file_hash)
                if upload_result:
                    break
                
                if attempt < self.max_retries:
                    self.logger.warning(f"上传失败，第 {attempt} 次重试: {file_path}")
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error(f"上传失败，已达到最大重试次数: {file_path}")
                    self.mark_failed(mark_file_path, file_hash, "上传失败")
                    return False
            
            # 获取文件URL
            file_info = upload_result.get('data', {}).get('file_info', {})
            file_url = file_info.get('file_url')
            
            if not file_url:
                self.logger.error(f"上传响应中未找到文件URL: {file_path}")
                self.mark_failed(mark_file_path, file_hash, "上传响应异常")
                return False
            
            # 重试机制：构建知识图谱
            build_success = False
            for attempt in range(1, self.max_retries + 1):
                build_success = self.call_build_api(file_url, file_hash)
                if build_success:
                    break
                
                if attempt < self.max_retries:
                    self.logger.warning(f"构建失败，第 {attempt} 次重试: {file_url}")
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error(f"构建失败，已达到最大重试次数: {file_url}")
                    self.mark_failed(mark_file_path, file_hash, "构建失败")
                    return False
            
            # 标记为已完成
            self.mark_processed(mark_file_path, file_hash)
            self.logger.info(f"文件处理成功: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"处理文件时发生异常 {file_path}: {e}", exc_info=True)
            self.mark_failed(mark_file_path, file_hash if 'file_hash' in locals() else "unknown", str(e))
            return False
        finally:
            # 从处理中集合移除
            with self.processing_lock:
                self.processing_files.discard(file_key)
    
    def create_processing_mark(self, mark_file_path: str, file_hash: str, status: str = "processing"):
        """
        创建处理标记文件
        
        Args:
            mark_file_path: 标记文件路径
            file_hash: 文件哈希值
            status: 处理状态（processing/completed/failed）
        """
        mark_data = {
            'file_hash': file_hash,
            'status': status,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        
        with open(mark_file_path, 'w', encoding='utf-8') as f:
            json.dump(mark_data, f, ensure_ascii=False, indent=2)
    
    def mark_processed(self, mark_file_path: str, file_hash: str):
        """
        更新标记文件为已完成状态
        
        Args:
            mark_file_path: 标记文件路径
            file_hash: 文件哈希值
        """
        mark_data = {
            'file_hash': file_hash,
            'status': 'completed',
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        
        with open(mark_file_path, 'w', encoding='utf-8') as f:
            json.dump(mark_data, f, ensure_ascii=False, indent=2)
    
    def mark_failed(self, mark_file_path: str, file_hash: str, error_message: str):
        """
        更新标记文件为失败状态
        
        Args:
            mark_file_path: 标记文件路径
            file_hash: 文件哈希值
            error_message: 错误信息
        """
        mark_data = {
            'file_hash': file_hash,
            'status': 'failed',
            'error': error_message,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        
        with open(mark_file_path, 'w', encoding='utf-8') as f:
            json.dump(mark_data, f, ensure_ascii=False, indent=2)


class FileMonitorHandler(FileSystemEventHandler):
    """文件系统事件处理器"""
    
    def __init__(self, 
                 processor: FileProcessor,
                 watch_directory: str,
                 supported_extensions: list,
                 batch_delay: float = 2.0,
                 batch_size: int = 5):
        """
        初始化文件监控处理器
        
        Args:
            processor: 文件处理器实例
            watch_directory: 监控目录
            supported_extensions: 支持的文件扩展名列表
            batch_delay: 批量处理延迟（秒），用于等待短时间内多个文件变化
            batch_size: 批量处理大小
        """
        self.processor = processor
        self.watch_directory = watch_directory
        self.supported_extensions = [ext.lower() for ext in supported_extensions]
        self.batch_delay = batch_delay
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)
        self.pending_files: Dict[str, float] = {}  # 待处理文件 {文件路径: 时间戳}
        self.executor = ThreadPoolExecutor(max_workers=3)  # 线程池处理文件
        self.batch_thread: Optional[Thread] = None
        self.batch_lock = Lock()
    
    def should_process_file(self, file_path: str) -> bool:
        """
        判断是否应该处理该文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            True表示应该处理，False表示跳过
        """
        # 检查文件是否存在
        if not os.path.exists(file_path):
            self.logger.debug(f"文件不存在: {file_path}")
            return False
        
        # 跳过目录
        if os.path.isdir(file_path):
            return False
        
        # 跳过隐藏文件
        if os.path.basename(file_path).startswith('.'):
            return False
        
        # 检查文件扩展名
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in self.supported_extensions:
            return False
        
        # 检查是否已存在标记文件
        mark_file_path = f"{file_path}.processed"
        if os.path.exists(mark_file_path):
            # 检查标记文件状态
            try:
                with open(mark_file_path, 'r', encoding='utf-8') as f:
                    mark_data = json.load(f)
                    status = mark_data.get('status', 'unknown')
                    
                    # 如果已完成，跳过
                    if status == 'completed':
                        self.logger.info(f"文件已处理完成，跳过: {file_path}")
                        return False
                    
                    # 如果正在处理中，跳过（避免重复处理）
                    if status == 'processing':
                        self.logger.info(f"文件正在处理中，跳过: {file_path}")
                        return False
                    
                    # 如果失败，允许重试（但需要手动触发或等待一段时间）
                    if status == 'failed':
                        self.logger.info(f"文件处理失败，可重试: {file_path}")
                        # 这里可以选择是否自动重试失败的文件
                        # 为了安全，默认不自动重试，需要手动删除标记文件
                        return False
            except Exception as e:
                self.logger.warning(f"读取标记文件失败 {mark_file_path}: {e}，将尝试重新处理")
                # 如果标记文件损坏，允许处理
        self.logger.debug(f"文件符合处理条件: {file_path}")
        return True
    
    def get_mark_file_path(self, file_path: str) -> str:
        """
        获取标记文件路径
        
        Args:
            file_path: 原文件路径
            
        Returns:
            标记文件路径
        """
        return f"{file_path}.processed"
    
    def process_file_async(self, file_path: str):
        """
        异步处理文件
        
        Args:
            file_path: 要处理的文件路径
        """
        mark_file_path = self.get_mark_file_path(file_path)
        
        # 提交到线程池处理
        future = self.executor.submit(self.processor.process_file, file_path, mark_file_path)
        
        def handle_result(future):
            try:
                success = future.result()
                if success:
                    self.logger.info(f"异步处理完成: {file_path}")
                else:
                    self.logger.error(f"异步处理失败: {file_path}")
            except Exception as e:
                self.logger.error(f"异步处理异常 {file_path}: {e}", exc_info=True)
        
        # 添加回调处理结果
        future.add_done_callback(handle_result)
    
    def batch_process_files(self):
        """批量处理待处理文件"""
        self.logger.info("批量处理线程已启动")
        while True:
            time.sleep(self.batch_delay)
            
            with self.batch_lock:
                if not self.pending_files:
                    continue
                
                # 获取待处理文件列表
                current_time = time.time()
                files_to_process = [
                    file_path for file_path, timestamp in self.pending_files.items()
                    if current_time - timestamp >= self.batch_delay
                ]
                
                if not files_to_process:
                    continue
                
                # 移除已到期的文件
                for file_path in files_to_process:
                    self.pending_files.pop(file_path, None)
            
            # 处理文件（限制批量大小）
            for file_path in files_to_process[:self.batch_size]:
                if self.should_process_file(file_path):
                    self.logger.info(f"批量处理文件: {file_path}")
                    self.process_file_async(file_path)
                else:
                    self.logger.debug(f"跳过文件（不符合处理条件）: {file_path}")
    
    def on_created(self, event: FileSystemEvent):
        """处理文件创建事件"""
        if not event.is_directory:
            file_path = event.src_path
            self.logger.info(f"检测到文件创建: {file_path}")
            
           # 等待文件完全写入（某些编辑器会先创建空文件再写入内容）
            time.sleep(0.5)
            
            # 检查文件是否真的存在且可读
            if not os.path.exists(file_path):
                self.logger.warning(f"文件创建事件后文件不存在: {file_path}")
                return
            
            # 检查文件大小，如果为0则等待一下
            try:
                file_size = os.path.getsize(file_path)
                if file_size == 0:
                    self.logger.debug(f"文件大小为0，等待写入完成: {file_path}")
                    time.sleep(1.0)
                    # 再次检查
                    if os.path.getsize(file_path) == 0:
                        self.logger.warning(f"文件大小仍为0，可能为空文件: {file_path}")
            except Exception as e:
                self.logger.warning(f"检查文件大小失败: {file_path}, {e}")
            
            # 快速检查是否符合处理条件
            if self.should_process_file(file_path):
                # 如果批量延迟很短（<=1秒），立即处理，不加入队列
                if self.batch_delay <= 1.0:
                    self.logger.info(f"批量延迟较短({self.batch_delay}秒)，立即处理文件: {file_path}")
                    self.process_file_async(file_path)
                else:
                    # 添加到待处理队列（批量处理）
                    with self.batch_lock:
                        self.pending_files[file_path] = time.time()
                        self.logger.info(f"文件已添加到待处理队列: {file_path}, 队列大小: {len(self.pending_files)}, 将在 {self.batch_delay} 秒后处理")
            else:
                self.logger.info(f"文件不符合处理条件，跳过: {file_path}")
    
    def on_modified(self, event: FileSystemEvent):
        """处理文件修改事件"""
        if not event.is_directory:
            file_path = event.src_path
            
            # 跳过标记文件本身
            if file_path.endswith('.processed'):
                return
            
            self.logger.info(f"检测到文件修改: {file_path}")
            
            # 检查是否应该处理
            if self.should_process_file(file_path):
                # 删除旧的标记文件（如果存在），允许重新处理
                mark_file_path = self.get_mark_file_path(file_path)
                if os.path.exists(mark_file_path):
                    try:
                        with open(mark_file_path, 'r', encoding='utf-8') as f:
                            mark_data = json.load(f)
                            old_hash = mark_data.get('file_hash', '')
                            
                            # 生成新哈希
                            new_hash = self.processor.generate_file_hash(file_path)
                            
                            # 如果文件内容未变化，跳过
                            if old_hash == new_hash:
                                self.logger.debug(f"文件内容未变化，跳过: {file_path}")
                                return
                            
                            # 文件内容已变化，删除旧标记
                            os.remove(mark_file_path)
                            self.logger.info(f"文件已修改，删除旧标记: {file_path}")
                    except Exception as e:
                        self.logger.warning(f"检查文件变化失败 {file_path}: {e}")
                
                # 添加到待处理队列
                with self.batch_lock:
                    self.pending_files[file_path] = time.time()


class KnowledgeGraphMonitor:
    """知识图谱文件监控主类"""
    
    def __init__(self, config_path: str = "config.json"):
        """
        初始化监控器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self.load_config()
        self.logger = setup_logging(self.config.get('log_dir'))
        
        # 初始化组件
        self.processor = FileProcessor(
            upload_api=self.config['upload_api'],
            build_api=self.config['build_graph_api'],
            max_retries=self.config.get('max_retries', 3),
            retry_delay=self.config.get('retry_delay', 5.0)
        )
        
        self.observer: Optional[Observer] = None
        self.handler: Optional[FileMonitorHandler] = None
    
    def load_config(self) -> Dict[str, Any]:
        """
        加载配置文件
        
        Returns:
            配置字典
        """
        default_config = {
            "watch_directory": "./wait_build",
            "upload_api": "http://dev-operate.moineye.com/graphapi/upload",
            "build_graph_api": "http://dev-operate.moineye.com/graphapi/build_graph_sse",
            "supported_extensions": [".txt", ".md", ".docx", ".pdf"],
            "log_dir": "./logs",
            "max_retries": 3,
            "retry_delay": 5.0,
            "batch_delay": 2.0,
            "batch_size": 5
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
                    print(f"已加载配置文件: {self.config_path}")
            except Exception as e:
                print(f"加载配置文件失败，使用默认配置: {e}")
        else:
            # 如果配置文件不存在，创建默认配置文件
            try:
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, ensure_ascii=False, indent=2)
                print(f"已创建默认配置文件: {self.config_path}")
            except Exception as e:
                print(f"创建配置文件失败: {e}")
        
        return default_config
    
    def initialize_watch_directory(self):
        """初始化监控目录"""
        watch_dir = self.config['watch_directory']
        
        if not os.path.exists(watch_dir):
            os.makedirs(watch_dir)
            self.logger.info(f"创建监控目录: {watch_dir}")
        else:
            self.logger.info(f"监控目录已存在: {watch_dir}")
    
    def start(self):
        """启动监控"""
        # 初始化监控目录
        self.initialize_watch_directory()
        
        # 创建事件处理器
        self.handler = FileMonitorHandler(
            processor=self.processor,
            watch_directory=self.config['watch_directory'],
            supported_extensions=self.config['supported_extensions'],
            batch_delay=self.config.get('batch_delay', 2.0),
            batch_size=self.config.get('batch_size', 5)
        )
        
        # 启动批量处理线程
        batch_thread = Thread(target=self.handler.batch_process_files, daemon=True,name="BatchProcessor")
        batch_thread.start()
        self.logger.info("批量处理线程已启动")
        # 创建观察者
        self.observer = Observer()
        self.observer.schedule(
            self.handler,
            self.config['watch_directory'],
            recursive=False  # 不递归监控子目录
        )
        
        # 启动观察者
        self.observer.start()
        self.logger.info(f"文件监控已启动，监控目录: {self.config['watch_directory']}")
        self.logger.info(f"支持的文件类型: {', '.join(self.config['supported_extensions'])}")
        self.logger.info(f"上传接口: {self.config['upload_api']}")
        self.logger.info(f"构建接口: {self.config['build_graph_api']}")
        
        try:
            # 保持运行
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """停止监控"""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.logger.info("文件监控已停止")
        
        if self.handler and self.handler.executor:
            self.handler.executor.shutdown(wait=True)
            self.logger.info("线程池已关闭")
    
    def retry_failed_files(self):
        """重试失败的文件（手动触发）"""
        watch_dir = self.config['watch_directory']
        retry_count = 0
        
        for file_name in os.listdir(watch_dir):
            file_path = os.path.join(watch_dir, file_name)
            
            # 跳过目录和标记文件
            if os.path.isdir(file_path) or file_name.endswith('.processed'):
                continue
            
            mark_file_path = f"{file_path}.processed"
            
            if os.path.exists(mark_file_path):
                try:
                    with open(mark_file_path, 'r', encoding='utf-8') as f:
                        mark_data = json.load(f)
                        status = mark_data.get('status', 'unknown')
                        
                        if status == 'failed':
                            # 删除失败标记，允许重试
                            os.remove(mark_file_path)
                            self.logger.info(f"准备重试失败文件: {file_path}")
                            
                            if self.handler and self.handler.should_process_file(file_path):
                                self.handler.process_file_async(file_path)
                                retry_count += 1
                except Exception as e:
                    self.logger.warning(f"检查失败文件时出错 {file_path}: {e}")
        
        self.logger.info(f"已触发 {retry_count} 个失败文件的重试")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='知识图谱文件监控自动化脚本')
    parser.add_argument(
        '--config',
        type=str,
        default='config.json',
        help='配置文件路径（默认: config.json）'
    )
    parser.add_argument(
        '--retry-failed',
        action='store_true',
        help='重试所有失败的文件'
    )
    
    args = parser.parse_args()
    
    # 创建监控器
    monitor = KnowledgeGraphMonitor(config_path=args.config)
    
    # 如果指定了重试失败文件
    if args.retry_failed:
        monitor.initialize_watch_directory()
        monitor.handler = FileMonitorHandler(
            processor=monitor.processor,
            watch_directory=monitor.config['watch_directory'],
            supported_extensions=monitor.config['supported_extensions'],
            batch_delay=monitor.config.get('batch_delay', 2.0),
            batch_size=monitor.config.get('batch_size', 5)
        )
        monitor.retry_failed_files()
        return
    
    # 启动监控
    monitor.start()


if __name__ == '__main__':
    main()
