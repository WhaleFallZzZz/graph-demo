#!/usr/bin/env python3
"""
增强的API服务器 - 支持SSE实时进度推送 + 简化的文件类型检测（仅扩展名和MIME类型）
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
from flask import Response, stream_with_context

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import logging
from kg_manager import builder, cos_uploader
from progress_sse import ProgressTracker, progress_manager, sse_event, create_progress_event, create_error_event, create_complete_event
from file_type_detector import file_detector, detect_file_type, is_allowed_file
from config import DOCUMENT_CONFIG, task_results, NEO4J_CONFIG

logger = logging.getLogger(__name__)

# 初始化构建器 (Gunicorn 启动时也会执行)
logger.info("正在初始化知识图谱构建器...")
if not builder.initialize():
    logger.error("构建器初始化失败")
    # 不在这里退出，以免影响其他 worker 或导致不断重启，但会记录严重错误

# 创建Flask应用
app = Flask(__name__)

# 配置
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB 文件大小限制

# 全局构建器实例
executor = ThreadPoolExecutor(max_workers=3)

def build_graph_with_progress(file_url: str, client_id: str) -> Dict[str, Any]:
    """带进度推送的知识图谱构建"""
    start_time = datetime.now()
    temp_dir = None
    
    try:
        # 创建进度跟踪器
        progress_tracker = ProgressTracker(client_id, total_steps=8)
        
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
            import requests
            
            # 下载文件
            response = requests.get(file_url, timeout=30)
            response.raise_for_status()
            
            # 获取文件名
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
                import mimetypes
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
        
        # 阶段5：完成
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
        # 恢复原始配置
        if 'original_path' in locals():
            DOCUMENT_CONFIG['path'] = original_path
        # 清理临时目录
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir)

@app.route('/upload', methods=['POST'])
def upload_file():
    """上传文件接口 - 返回JSON"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': '没有文件上传'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
            
        # 获取参数
        max_size = request.form.get('max_size', type=int) or (200 * 1024 * 1024)
        
        # 读取文件数据
        file_data = file.read()
        # 使用原始文件名（仅保留基本名称，避免路径遍历），解决中文文件名被 secure_filename 过滤的问题
        filename = os.path.basename(file.filename)
        
        # 创建临时文件进行类型检测
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file_data)
            temp_file_path = temp_file.name
        
        try:
            # 文件类型检测
            detection_result = detect_file_type(temp_file_path, file.filename)
            
            if 'error' in detection_result:
                return jsonify({'error': f"文件检测失败: {detection_result['error']}"}), 400
            if not detection_result.get('allowed', False):
                detected_type = detection_result.get('detected', '未知')
                return jsonify({'error': f"不支持的文件类型: {detected_type}"}), 400
            
            file_type = detection_result.get('type', '未知')
            
        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        
        # 检查文件大小
        if len(file_data) > max_size:
            return jsonify({'error': f'文件大小不能超过 {max_size // 1024 // 1024}MB'}), 400
        
        # 上传到COS
        if not cos_uploader:
            return jsonify({'error': 'COS上传器未初始化'}), 500
        
        upload_result = cos_uploader.upload_file(file_data, filename, max_size)
        
        # 返回成功结果
        return jsonify({
            'success': True,
            'message': '文件上传成功',
            'data': {
                'file_info': {
                    'filename': filename,
                    'file_url': upload_result['file_url'],
                    'size': upload_result['size'],
                    'md5': upload_result['md5'],
                    'file_type': file_type
                }
            }
        })
        
    except Exception as e:
        logger.error(f"上传接口出错: {e}")
        return jsonify({'error': f'上传失败: {str(e)}'}), 500

@app.route('/build_graph_sse', methods=['POST'])
def build_graph_sse():
    """根据文件URL构建图谱 - 返回SSE"""
    try:
        data = request.json or request.form
        file_url = data.get('file_url')
        
        logger.info(f"收到构建请求 build_graph_sse: {data}")
        
        if not file_url:
             return Response(sse_event(create_error_event("validation", "缺少 file_url 参数")),
                             mimetype='text/event-stream'), 400

        client_id = f"client_{int(datetime.now().timestamp() * 1000)}"
        
        def generate_events():
            """生成SSE事件流"""
            import queue
            
            q = queue.Queue()
            def progress_callback(data):
                q.put(data)
            
            progress_manager.add_listener(client_id, progress_callback)
            
            try:
                # 初始进度
                yield sse_event(create_progress_event("knowledge_graph", "开始构建知识图谱...", 0))
                
                # 提交任务
                future = executor.submit(build_graph_with_progress, file_url, client_id)
                
                while True:
                    try:
                        data = q.get(timeout=1.0)
                        yield sse_event(data)
                        if data.get('type') in ['complete', 'error']:
                            break
                    except queue.Empty:
                        if future.done():
                            try:
                                exception = future.exception()
                                if exception:
                                    logger.error(f"后台任务异常: {exception}")
                                    yield sse_event(create_error_event("unknown", f"后台处理异常: {str(exception)}"))
                                break
                            except Exception:
                                break
                        yield ": heartbeat\n\n"
                        continue
            except Exception as e:
                logger.error(f"SSE处理出错: {e}")
                yield sse_event(create_error_event("unknown", str(e)))
            finally:
                progress_manager.remove_listener(client_id)
                
        return Response(stream_with_context(generate_events()), 
                        mimetype='text/event-stream',
                        headers={
                            'Cache-Control': 'no-cache',
                            'X-Accel-Buffering': 'no',
                            'Access-Control-Allow-Origin': '*'
                        })
        
    except Exception as e:
        logger.error(f"构建接口出错: {e}")
        return Response(sse_event(create_error_event("unknown", str(e))), mimetype='text/event-stream'), 500

@app.route('/upload_and_build_sse', methods=['POST'])
def upload_and_build_sse():
    """上传到腾讯云COS并构建知识图谱（增强版SSE，带详细进度和简化文件类型检测）"""
    try:
        if 'file' not in request.files:
            return Response(sse_event(create_error_event("validation", "没有文件上传")),
                          mimetype='text/event-stream'), 400
        
        file = request.files['file']
        if file.filename == '':
            return Response(sse_event(create_error_event("validation", "没有选择文件")),
                          mimetype='text/event-stream'), 400
        
        # 生成客户端ID
        client_id = f"client_{int(datetime.now().timestamp() * 1000)}"
        
        # 获取参数
        max_size = request.form.get('max_size', type=int) or (200 * 1024 * 1024)  # 默认200MB
        
        # 读取文件数据
        file_data = file.read()
        # 使用原始文件名（仅保留基本名称，避免路径遍历），解决中文文件名被 secure_filename 过滤的问题
        filename = os.path.basename(file.filename)
        
        def generate_events():
            """生成SSE事件流 (使用队列+线程实现实时推送)"""
            import queue
            import threading
            
            # 创建消息队列
            q = queue.Queue()
            
            # 定义回调函数
            def progress_callback(data):
                q.put(data)
            
            # 注册监听器
            progress_manager.add_listener(client_id, progress_callback)
            
            try:
                # 阶段1：文件验证
                yield sse_event(create_progress_event("validation", "正在验证文件...", 5))
                
                # ... (文件验证逻辑保持不变) ...
                # 创建临时文件进行类型检测
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(file_data)
                    temp_file_path = temp_file.name
                
                try:
                    # 使用简化的文件类型检测
                    detection_result = detect_file_type(temp_file_path, file.filename)
                    
                    if 'error' in detection_result:
                        yield sse_event(create_error_event("validation", f"文件检测失败: {detection_result['error']}"))
                        return
                    
                    if not detection_result.get('allowed', False):
                        detected_type = detection_result.get('detected', '未知')
                        yield sse_event(create_error_event("validation", f"不支持的文件类型: {detected_type}。支持的类型: txt, docx, pdf, html, md, py, json, xml, yaml"))
                        return
                    
                    # 记录检测详情
                    file_type = detection_result.get('type', '未知')
                    confidence = detection_result.get('confidence', 0)
                    methods = detection_result.get('method', [])
                    
                    yield sse_event(create_progress_event("validation", f"文件类型检测完成: {file_type} (置信度: {confidence}, 方法: {', '.join(methods)})", 10))
                    
                finally:
                    # 清理临时文件
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
                
                # 检查文件大小
                if len(file_data) > max_size:
                    yield sse_event(create_error_event("validation", f'文件大小不能超过 {max_size // 1024 // 1024}MB'))
                    return
                
                yield sse_event(create_progress_event("validation", f"文件验证通过: {filename} ({len(file_data)} bytes)", 15))
                
                # 阶段2：文件上传
                yield sse_event(create_progress_event("upload", "正在上传到腾讯云COS...", 20))
                
                # 检查COS上传器
                if not cos_uploader:
                    yield sse_event(create_error_event("upload", 'COS上传器未初始化'))
                    return
                
                # 执行上传 (这里保持同步，因为上传通常很快且也是阻塞IO)
                upload_result = cos_uploader.upload_file(file_data, filename, max_size)
                
                yield sse_event(create_progress_event("upload", "文件上传成功", 30, {
                    'file_info': {
                        'filename': filename,
                        'file_url': upload_result['file_url'],
                        'size': upload_result['size'],
                        'md5': upload_result['md5'],
                        'file_type': file_type
                    }
                }))
                
                # 阶段3：知识图谱构建（这是耗时操作，放入后台线程）
                yield sse_event(create_progress_event("knowledge_graph", "开始构建知识图谱...", 40))
                
                # 在后台线程启动构建任务
                # 注意：build_graph_with_progress 内部会通过 ProgressTracker -> progress_manager -> callback -> queue 发送进度
                future = executor.submit(build_graph_with_progress, upload_result['file_url'], client_id)
                
                # 循环从队列读取进度并推送到SSE流
                while True:
                    try:
                        # 阻塞等待消息，设置超时防止死锁
                        # 这里的超时也是一种心跳机制，确保连接活跃
                        data = q.get(timeout=1.0) 
                        yield sse_event(data)
                        
                        # 检查是否完成或出错
                        msg_type = data.get('type')
                        if msg_type in ['complete', 'error']:
                            break
                            
                    except queue.Empty:
                        # 队列空闲，检查任务是否已完成（防止回调丢失导致的死循环）
                        if future.done():
                            # 任务已结束但队列空了，说明可能最后一条消息已处理或异常退出
                            # 这里可以检查 future.result() 或 future.exception()
                            try:
                                # 如果任务抛出未捕获异常，这里会重新抛出
                                exception = future.exception()
                                if exception:
                                    logger.error(f"后台任务异常: {exception}")
                                    yield sse_event(create_error_event("unknown", f"后台处理发生异常: {str(exception)}"))
                                break
                            except Exception as e:
                                logger.error(f"检查后台任务状态出错: {e}")
                                break
                        
                        # 发送心跳注释，防止网关/浏览器超时
                        yield ": heartbeat\n\n"
                        continue
                        
            except Exception as e:
                logger.error(f"SSE处理过程出错: {e}")
                yield sse_event(create_error_event("unknown", f'处理过程出错: {str(e)}'))
            finally:
                # 清理监听器
                progress_manager.remove_listener(client_id)
        
        # 返回SSE流
        return Response(
            stream_with_context(generate_events()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',  # 禁用Nginx缓冲
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Cache-Control'
            }
        )
        
    except Exception as e:
        logger.error(f"SSE接口处理出错: {e}")
        return Response(sse_event(create_error_event("unknown", f'接口处理出错: {str(e)}')),
                      mimetype='text/event-stream'), 500

# 其他接口保持不变...
@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'knowledge-graph-api'
    })

@app.route('/task_status/<task_id>', methods=['GET'])
def task_status(task_id: str):
    """查询任务状态"""
    try:
        if task_id in task_results:
            result = task_results[task_id]
            return jsonify({
                'task_id': task_id,
                'status': result['status'],
                'graph_id': result['graph_id'],
                'entities_count': result['entities_count'],
                'relationships_count': result['relationships_count'],
                'created_at': result['created_at'],
                'completed_at': result['completed_at']
            })
        else:
            return jsonify({'error': '任务不存在'}), 404
    except Exception as e:
        logger.error(f"查询任务状态失败: {e}")
        return jsonify({'error': f'查询失败: {e}'}), 500

@app.route('/graph/data', methods=['GET'])
def get_graph_data():
    """获取Neo4j中的节点和边数据 - 复用构建器的图存储连接"""
    try:
        nodes = []
        edges = []
        
        # 复用构建器中的图存储连接
        if not builder.graph_store:
            if not builder.initialize():
                return jsonify({
                    "code": 500,
                    "msg": "图存储未初始化",
                    "data": {"nodes": [], "edges": []}
                }), 500
        
        # 使用 structured_query 执行 Cypher 查询
        # 获取所有节点（排除embedding相关的节点）
        node_query = """
            MATCH (n) 
            WHERE NOT n:Embedding AND NOT n:__Embedding__ AND NOT n:__Vector__
            RETURN n, elementId(n) as id 
            LIMIT 1000
        """
        node_result = builder.graph_store.structured_query(node_query)
        
        for record in node_result:
            node = record.get("n", {})
            node_id = record.get("id", "")
            
            # 获取节点标签和属性
            node_labels = node.get("labels", []) if isinstance(node, dict) else getattr(node, 'labels', [])
            node_props = dict(node) if hasattr(node, 'items') else {}
            
            # 过滤掉embedding属性
            node_properties = {
                k: v for k, v in node_props.items() 
                if not k.startswith('embedding') and not k.startswith('__')
            }
            
            label = list(node_labels)[0] if node_labels else "Unknown"
            
            nodes.append({
                "id": f"node_{node_id}",
                "label": label,
                "type": "circle",
                "data": {"category": label, **node_properties}
            })
        
        # 获取所有关系
        rel_query = """
            MATCH (a)-[r]->(b) 
            WHERE NOT a:Embedding AND NOT a:__Embedding__ AND NOT a:__Vector__
              AND NOT b:Embedding AND NOT b:__Embedding__ AND NOT b:__Vector__
              AND NOT type(r) CONTAINS 'EMBEDDING' AND NOT type(r) CONTAINS 'embedding'
            RETURN 
                elementId(a) as source_id,
                elementId(b) as target_id,
                type(r) as rel_type,
                properties(r) as props
            LIMIT 1000
        """
        rel_result = builder.graph_store.structured_query(rel_query)
        
        for record in rel_result:
            edges.append({
                "source": f"node_{record.get('source_id', '')}",
                "target": f"node_{record.get('target_id', '')}",
                "label": record.get("rel_type", ""),
                "data": dict(record.get("props", {})) if record.get("props") else {}
            })
        
        logger.info(f"成功获取图数据: {len(nodes)} 个节点, {len(edges)} 条边")
        
        return jsonify({
            "code": 200,
            "msg": "success",
            "data": {"nodes": nodes, "edges": edges}
        })
        
    except Exception as e:
        logger.error(f"获取图数据失败: {e}")
        return jsonify({
            "code": 500,
            "msg": f"获取图数据失败: {str(e)}",
            "data": {"nodes": [], "edges": []}
        }), 500

@app.route('/search', methods=['GET', 'POST'])
def search_knowledge_graph():
    """根据传参msg来检索知识图谱"""
    try:
        # 获取查询参数
        msg = request.args.get('msg')
        if not msg and request.is_json:
            msg = request.json.get('msg')
            
        if not msg:
            return jsonify({
                'code': 400, 
                'msg': '缺少msg参数', 
                'data': None
            }), 400
            
        # 确保构建器已初始化
        if not builder.llm or not builder.graph_store:
            # 尝试重新初始化
            logger.info("构建器组件未就绪，尝试初始化...")
            builder.initialize()
            
        # 执行查询
        logger.info(f"收到搜索请求: {msg}")
        result = builder.query_knowledge_graph(msg)
        
        return jsonify({
            'code': 200, 
            'msg': 'success', 
            'data': {
                'answer': result,
                'query': msg
            }
        })
    except Exception as e:
        logger.error(f"搜索接口出错: {e}")
        return jsonify({
            'code': 500, 
            'msg': f"搜索失败: {str(e)}", 
            'data': None
        }), 500

if __name__ == '__main__':
    # 初始化构建器
    logger.info("正在初始化知识图谱构建器...")
    if not builder.initialize():
        logger.error("构建器初始化失败，服务启动中止")
        sys.exit(1)
    
    logger.info("✅ 知识图谱API服务器启动成功")
    logger.info("🚀 服务运行在 http://localhost:8001")
    logger.info("📊 健康检查: GET /health")
    logger.info("📤 文件上传: POST /upload_and_build_sse")
    logger.info("📋 任务查询: GET /task_status/<task_id>")
    logger.info("🕸️ 图数据: GET /graph/data")
    
    # 启动Flask应用
    app.run(host='0.0.0.0', port=8001, debug=False, threaded=True)
