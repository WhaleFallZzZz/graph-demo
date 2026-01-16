#!/usr/bin/env python3
"""
图谱构建服务层 - 封装知识图谱构建的核心业务逻辑
"""

import os
import sys
import shutil
import tempfile
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor

import requests
import certifi
import mimetypes

# 确保路径正确
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from llama.kg_manager import builder
from llama.progress_sse import ProgressTracker, progress_manager
from llama.config import DOCUMENT_CONFIG, task_results, RATE_LIMIT_CONFIG
from llama.common.dynamic_resource_allocator import DynamicScalingManager, ResourceAllocation

logger = logging.getLogger(__name__)

class GraphService:
    """知识图谱构建服务"""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.scaling_manager: Optional[DynamicScalingManager] = None
        self._initialize_builder()
        self._initialize_dynamic_scaling()

    def _initialize_builder(self):
        """初始化构建器"""
        logger.info("正在初始化知识图谱构建器...")
        if not builder.initialize():
            logger.error("构建器初始化失败")
            # 不在这里退出，以免影响其他 worker，但会记录严重错误

    def _initialize_dynamic_scaling(self):
        """初始化动态资源分配系统"""
        try:
            # 获取当前Worker ID（从环境变量或进程ID）
            worker_id = os.getenv('WORKER_ID', f"worker_{os.getpid()}")
            
            # 获取总Worker数量
            total_workers = int(os.getenv('WORKER_COUNT', '4'))
            
            # 创建基础资源分配配置
            base_allocation = ResourceAllocation(
                max_concurrent_requests=RATE_LIMIT_CONFIG.get('max_concurrent_requests', 5),
                rpm_limit=RATE_LIMIT_CONFIG.get('rpm_limit', 60),
                tpm_limit=RATE_LIMIT_CONFIG.get('tpm_limit', 100000),
                num_workers=3  # executor的max_workers
            )
            
            # 创建动态缩放管理器
            self.scaling_manager = DynamicScalingManager(
                worker_id=worker_id,
                total_workers=total_workers,
                base_allocation=base_allocation,
                enable_scaling=True
            )
            
            # 设置资源分配回调函数
            self.scaling_manager.set_allocation_callback(self.apply_resource_allocation)
            
            # 启动监控和调整
            self.scaling_manager.start()
            
            logger.info(
                f"✅ 动态资源分配系统已启动: "
                f"worker_id={worker_id}, total_workers={total_workers}"
            )
            
        except Exception as e:
            logger.error(f"初始化动态资源分配系统失败: {e}")
            self.scaling_manager = None

    def apply_resource_allocation(self, allocation: ResourceAllocation):
        """
        应用资源分配到配置
        
        Args:
            allocation: 资源分配配置
        """
        try:
            # 更新速率限制配置
            # 注意：这里直接修改导入的字典对象
            RATE_LIMIT_CONFIG['max_concurrent_requests'] = allocation.max_concurrent_requests
            RATE_LIMIT_CONFIG['rpm_limit'] = allocation.rpm_limit
            RATE_LIMIT_CONFIG['tpm_limit'] = allocation.tpm_limit
            
            # TODO: 动态调整线程池大小 (需要重建 executor，比较复杂，暂时忽略)
            # if allocation.num_workers != self.executor._max_workers:
            #     pass
            
            logger.info(
                f"✅ 资源分配已应用: "
                f"concurrent={allocation.max_concurrent_requests}, "
                f"rpm={allocation.rpm_limit}, "
                f"tpm={allocation.tpm_limit}, "
                f"workers={allocation.num_workers}"
            )
        except Exception as e:
            logger.error(f"应用资源分配失败: {e}")

    def _run_node_cleaning(self):
        """运行节点清洗脚本"""
        try:
            import subprocess
            import sys
            script_path = Path(__file__).parent.parent / "scripts" / "offline_node_cleaning.py"
            logger.info(f"开始执行节点清洗: {script_path}")
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode != 0:
                logger.warning(f"节点清洗执行失败: {result.stderr}")
            else:
                logger.info(f"节点清洗执行成功")
        except Exception as e:
            logger.error(f"节点清洗执行出错: {e}")

    def _run_entity_alignment(self):
        """运行实体对齐脚本"""
        try:
            import subprocess
            import sys
            script_path = Path(__file__).parent.parent / "scripts" / "offline_entity_alignment.py"
            logger.info(f"开始执行实体对齐: {script_path}")
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode != 0:
                logger.warning(f"实体对齐执行失败: {result.stderr}")
            else:
                logger.info(f"实体对齐执行成功")
        except Exception as e:
            logger.error(f"实体对齐执行出错: {e}")

    def _run_property_sinking(self):
        """运行属性下沉脚本"""
        try:
            import subprocess
            import sys
            script_path = Path(__file__).parent.parent / "scripts" / "offline_property_sinking.py"
            logger.info(f"开始执行属性下沉: {script_path}")
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode != 0:
                logger.warning(f"属性下沉执行失败: {result.stderr}")
            else:
                logger.info(f"属性下沉执行成功")
        except Exception as e:
            logger.error(f"属性下沉执行出错: {e}")

    def submit_build_task(self, file_url: str, client_id: str, custom_file_name: Optional[str] = None):
        """提交构建任务到线程池"""
        return self.executor.submit(self.build_graph_with_progress, file_url, client_id, custom_file_name)

    def build_graph_with_progress(self, file_url: str, client_id: str, custom_file_name: Optional[str] = None) -> Dict[str, Any]:
        """带进度推送的知识图谱构建
        
        Args:
            file_url: 文件URL
            client_id: 客户端ID
            custom_file_name: 自定义文件名（可选），如果提供则使用该名称而非从URL中提取
        """
        start_time = datetime.now()
        temp_dir = None
        
        # 标记Worker为活跃状态
        if self.scaling_manager:
            self.scaling_manager.update_activity(is_active=True, current_load=1.0, active_tasks=1)
        
        try:
            # 创建进度跟踪器
            progress_tracker = ProgressTracker(client_id, total_steps=9)
            
            # 阶段1：初始化
            progress_tracker.update_stage("initialization", "正在初始化构建器...", 10)
            
            # 检查构建器是否初始化
            if not builder:
                error_msg = "知识图谱构建器未初始化"
                progress_tracker.error("initialization", error_msg)
                return {'success': False, 'error': error_msg}
            
            # 阶段2：下载文件
            progress_tracker.update_stage("file_download", "正在下载文件...", 20)
            
            # 创建临时目录用于文档处理
            temp_dir = Path(tempfile.mkdtemp())
            
            # 从COS URL下载文件
            if file_url.startswith('https://') and '.cos.' in file_url:
                try:
                    response = requests.get(file_url, timeout=30, verify=certifi.where())
                except requests.exceptions.SSLError as ssl_err:
                    logger.info(f"COS证书验证失败，降级为不验证: {ssl_err}")
                    response = requests.get(file_url, timeout=30, verify=False)
                response.raise_for_status()
                
                # 获取文件名
                # 优先使用自定义文件名，否则从URL提取
                if custom_file_name:
                    filename = custom_file_name
                    logger.info(f"使用自定义文件名: {filename}")
                else:
                    filename = file_url.split('/')[-1].split('?')[0]
                
                # 尝试修复文件名后缀
                # 1. 如果文件名以 _pdf, _docx 等结尾，替换为 .pdf, .docx (针对特殊OSS/COS链接)
                if filename.endswith('_pdf'):
                    filename = filename[:-4] + '.pdf'
                elif filename.endswith('_docx'):
                    filename = filename[:-5] + '.docx'
                elif filename.endswith('_txt'):
                    filename = filename[:-4] + '.txt'
                
                # 2. 如果没有后缀，尝试从Content-Type推断
                if not Path(filename).suffix:
                    content_type = response.headers.get('Content-Type')
                    if content_type:
                        ext = mimetypes.guess_extension(content_type)
                        if ext:
                            # mimetypes.guess_extension 可能返回 .jpe 而不是 .jpg，但在我们的场景下主要是 pdf/docx
                            filename = filename + ext
                
                temp_file = temp_dir / filename
                
                # 保存文件
                with open(temp_file, 'wb') as f:
                    f.write(response.content)
                    
                logger.info(f"从COS下载文件成功: {filename}")
            else:
                error_msg = '只支持腾讯云COS文件URL'
                progress_tracker.error("file_download", error_msg)
                return {'success': False, 'error': error_msg}
            
            # 阶段3：加载文档
            progress_tracker.update_stage("document_loading", "正在加载文档...", 30)
            
            # 临时修改DOCUMENT_CONFIG路径
            original_path = DOCUMENT_CONFIG['path']
            DOCUMENT_CONFIG['path'] = str(temp_dir)
            
            # 加载文档
            documents = builder.load_documents(progress_tracker)
            if not documents:
                error_msg = '无法加载文档'
                progress_tracker.error("document_loading", error_msg)
                return {'success': False, 'error': error_msg}
            
            # 将自定义文件名添加到所有文档的metadata中
            for doc in documents:
                if not hasattr(doc, 'metadata'):
                    doc.metadata = {}
                # 保存原始文件名信息
                # doc.metadata['source_file_name'] = filename
                doc.metadata['created_at'] = int(datetime.now().timestamp())
                doc.metadata['updated_at'] = 0
                doc.metadata['deleted_at'] = 0
                doc.metadata['source'] = 'system'
                doc.metadata['file_url'] = file_url
                logger.debug(f"文档metadata已更新: source_file_name={filename}")
            
            # 阶段4：构建知识图谱
            progress_tracker.update_stage("knowledge_graph", "开始构建知识图谱...", 40)
            
            # 预检: 检查llm_outputs目录权限
            llm_outputs_dir = Path(os.getcwd()) / "llm_outputs"
            try:
                if not llm_outputs_dir.exists():
                    llm_outputs_dir.mkdir(parents=True, exist_ok=True)
                    logger.info(f"已创建输出目录: {llm_outputs_dir}")
                
                # 检查写权限
                test_file = llm_outputs_dir / ".test_write"
                with open(test_file, 'w') as f:
                    f.write('test')
                test_file.unlink()
                logger.info(f"输出目录权限检查通过: {llm_outputs_dir}")
            except Exception as e:
                logger.error(f"输出目录权限检查失败: {e}")
                # 不阻断流程，但记录警告
            
            logger.info(f"开始调用 builder.build_knowledge_graph, 文档数: {len(documents)}")
            
            # 构建知识图谱
            index = builder.build_knowledge_graph(documents, progress_tracker)
            
            if not index:
                error_msg = '知识图谱构建失败'
                progress_tracker.error("knowledge_graph", error_msg)
                return {'success': False, 'error': error_msg}
                
            logger.info("builder.build_knowledge_graph 调用成功")
            
            # 阶段5：后处理 - 节点清洗
            progress_tracker.update_stage("postprocessing_node_cleaning", "正在执行节点清洗...", 50)
            self._run_node_cleaning()
            
            # todo 有 bug 先注释掉，阶段6：后处理 - 实体对齐
            # progress_tracker.update_stage("postprocessing_entity_alignment", "正在执行实体对齐...", 60)
            # self._run_entity_alignment()
            
            # 阶段7：后处理 - 属性下沉
            progress_tracker.update_stage("postprocessing_property_sinking", "正在执行属性下沉...", 70)
            self._run_property_sinking()
            
            # 阶段8：完成
            processing_time = (datetime.now() - start_time).total_seconds()
            task_id = f"task_{int(start_time.timestamp())}"
            
            # 存储任务结果
            task_results[task_id] = {
                'status': 'completed',
                'graph_id': f"graph_{int(start_time.timestamp())}",
                'entities_count': len(documents) * 5,  # 估算
                'relationships_count': len(documents) * 10,  # 估算
                'created_at': start_time.isoformat(),
                'completed_at': datetime.now().isoformat()
            }
            
            # 完成结果
            result = {
                'success': True,
                'task_id': task_id,
                'graph_id': f"graph_{int(start_time.timestamp())}",
                'document_count': len(documents),
                'processing_time': processing_time,
                'file_info': {
                    'filename': filename,
                    'file_url': file_url
                }
            }
            
            progress_tracker.complete(result)
            return result
            
        except Exception as e:
            error_msg = f"知识图谱构建过程失败: {e}"
            logger.error(error_msg)
            
            if 'progress_tracker' in locals():
                progress_tracker.error("knowledge_graph", error_msg)
            
            return {'success': False, 'error': error_msg}
            
        finally:
            # 标记Worker为非活跃状态
            if self.scaling_manager:
                self.scaling_manager.update_activity(is_active=False, current_load=0.0, active_tasks=0)
            
            # 恢复原始配置
            if 'original_path' in locals():
                DOCUMENT_CONFIG['path'] = original_path
            # 清理临时目录
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir)

# 全局单例
graph_service = GraphService()
