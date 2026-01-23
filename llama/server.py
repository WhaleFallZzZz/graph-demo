#!/usr/bin/env python3
"""
增强的API服务器 - 支持SSE实时进度推送 + 简化的文件类型检测（仅扩展名和MIME类型）
"""

import os
import sys
import tempfile
import queue
from pathlib import Path
from datetime import datetime
from flask import Response, stream_with_context
from flask_cors import CORS

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from flask import Flask, request, jsonify
import logging
from llama.kg_manager import builder
from llama.config import cos_uploader
from llama.progress_sse import progress_manager, sse_event, create_progress_event, create_error_event, consume_sse_queue
from llama.file_type_detector import detect_file_type
from llama.graph_service import graph_service
from llama.query_preprocessor import QueryPreprocessor
from llama.hard_match_postprocessor import HardMatchPostprocessor

logger = logging.getLogger(__name__)

# 创建Flask应用
app = Flask(__name__)
CORS(app)  # 启用跨域资源共享

# 配置
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB 文件大小限制

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
    """根据文件URL构建图谱 - 返回SSE
    
    请求参数：
        - file_url: 文件URL（必选）
        - file_name: 自定义文件名（可选），如果不提供则从URL中提取
    
    使用示例：
        # 使用自定义文件名
        POST /build_graph_sse
        Content-Type: application/json
        {
            "file_url": "https://example.cos.ap-beijing.myqcloud.com/document.pdf",
            "file_name": "青少年近视防控手册.pdf"
        }
        
        # 不提供 file_name，使用URL中的文件名
        POST /build_graph_sse
        Content-Type: application/json
        {
            "file_url": "https://example.cos.ap-beijing.myqcloud.com/document.pdf"
        }
    """
    try:
        # 使用 silent=True 避免 JSON 解析失败时直接抛出 400 错误
        # 尝试获取 JSON 数据
        json_data = request.get_json(silent=True)
        # 尝试获取表单数据
        form_data = request.form
        
        # 合并数据 (优先使用 JSON)
        data = {}
        if form_data:
            data.update(form_data.to_dict())
        if json_data:
            data.update(json_data)
            
        file_url = data.get('file_url')
        custom_file_name = data.get('file_name')  # 新增可选参数
        
        # 记录原始请求数据以便调试
        if not data:
            logger.warning(f"收到 build_graph_sse 请求但无法解析数据. Content-Type: {request.content_type}")
            try:
                logger.debug(f"Raw data: {request.get_data(as_text=True)[:1000]}")
            except:
                pass
        else:
            logger.info(f"收到构建请求 build_graph_sse: {data}")
        
        if not file_url:
             return Response(sse_event(create_error_event("validation", "缺少 file_url 参数")),
                             mimetype='text/event-stream'), 400

        client_id = f"client_{int(datetime.now().timestamp() * 1000)}"
        
        def generate_events():
            """生成SSE事件流"""
            
            q = queue.Queue()
            def progress_callback(data):
                q.put(data)
            
            progress_manager.add_listener(client_id, progress_callback)
            
            try:
                # 初始进度
                yield sse_event(create_progress_event("knowledge_graph", "开始构建知识图谱...", 0))
                
                # 提交任务，传递custom_file_name参数
                future = graph_service.submit_build_task(file_url, client_id, custom_file_name)
                
                def check_future():
                    if future.done():
                        try:
                            exception = future.exception()
                            if exception:
                                logger.error(f"后台任务异常: {exception}")
                                q.put(create_error_event("unknown", f"后台处理异常: {str(exception)}"))
                                return False 
                        except Exception:
                            pass
                        return True
                    return False
                    
                yield from consume_sse_queue(q, check_future)
                
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
                          mimetype='text/event-stream',
                          headers={'Access-Control-Allow-Origin': '*'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return Response(sse_event(create_error_event("validation", "没有选择文件")),
                          mimetype='text/event-stream',
                          headers={'Access-Control-Allow-Origin': '*'}), 400
        
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
                
                # 在后台线程启动构建任务，传递filename作为custom_file_name
                # 注意：build_graph_with_progress 内部会通过 ProgressTracker -> progress_manager -> callback -> queue 发送进度
                future = graph_service.submit_build_task(upload_result['file_url'], client_id, filename)
                
                def check_future():
                    if future.done():
                        try:
                            # 如果任务抛出未捕获异常，这里会重新抛出
                            exception = future.exception()
                            if exception:
                                logger.error(f"后台任务异常: {exception}")
                                q.put(create_error_event("unknown", f"后台处理发生异常: {str(exception)}"))
                                return False
                        except Exception as e:
                            logger.error(f"检查后台任务状态出错: {e}")
                        return True
                    return False

                yield from consume_sse_queue(q, check_future)
                        
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

@app.route('/scaling_status', methods=['GET'])
def scaling_status():
    """动态资源分配状态接口"""
    try:
        if graph_service.scaling_manager is None:
            return jsonify({
                'error': '动态资源分配系统未初始化'
            }), 503
        
        status = graph_service.scaling_manager.get_status()
        return jsonify(status)
    except Exception as e:
        logger.error(f"获取缩放状态失败: {e}")
        return jsonify({
            'error': f'获取缩放状态失败: {str(e)}'
        }), 500

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

@app.route('/nodes/generate_embeddings', methods=['POST'])
def generate_node_embeddings():
    """为节点生成 embedding 向量接口"""
    try:
        data = request.json or {}
        node_ids = data.get('node_ids', [])
        node_names = data.get('node_names', [])
        
        # 确保构建器已初始化
        if not builder.embed_model or not builder.graph_store:
            if not builder.initialize():
                return jsonify({
                    'code': 500,
                    'msg': '构建器组件未就绪',
                    'data': None
                }), 500
        
        # 调用生成 embedding 方法
        result = builder.generate_embeddings_for_nodes(
            node_ids=node_ids if node_ids else None,
            node_names=node_names if node_names else None
        )
        
        if result['success']:
            return jsonify({
                'code': 200,
                'msg': result['message'],
                'data': {
                    'processed': result['processed'],
                    'failed': result['failed'],
                    'failed_nodes': result.get('failed_nodes')
                }
            })
        else:
            return jsonify({
                'code': 500,
                'msg': result['message'],
                'data': {
                    'processed': result.get('processed', 0),
                    'failed': result.get('failed', 0)
                }
            }), 500
            
    except Exception as e:
        logger.error(f"生成节点 embedding 接口出错: {e}")
        return jsonify({
            'code': 500,
            'msg': f"生成 embedding 失败: {str(e)}",
            'data': None
        }), 500

@app.route('/search', methods=['GET', 'POST'])
def search_knowledge_graph():
    """根据传参msg来检索知识图谱
    固定以SSE流式输出
    """
    try:
        msg = request.args.get('msg')
        if not msg and request.is_json:
            json_data = request.json
            msg = json_data.get('msg')
        
        if not msg:
            return Response(sse_event(create_error_event("search", "缺少msg参数")), mimetype='text/event-stream'), 400
            
        # 确保构建器已初始化
        if not builder.llm or not builder.graph_store:
            logger.info("构建器组件未就绪，尝试初始化...")
            builder.initialize()
        
        logger.info(f"收到流式搜索请求: {msg}")
        
        # 查询前置处理：意图分析、查询改写、实体硬匹配
        preprocess_result = None
        try:
            if builder.graph_agent and builder.graph_store:
                preprocessor = QueryPreprocessor(builder.graph_agent, builder.graph_store)
                preprocess_result = preprocessor.preprocess(msg)
                logger.info(f"查询前置处理完成: 意图={preprocess_result['intent']}, "
                          f"提取实体={len(preprocess_result['extracted_entities'])}, "
                          f"硬匹配实体={len(preprocess_result['hard_match_entities'])}")
        except Exception as e:
            logger.warning(f"查询前置处理失败，使用原始查询: {e}")
        
        def generate():
            # 如果前置处理成功，使用增强后的查询和硬匹配节点
            if preprocess_result:
                enhanced_query = preprocess_result['enhanced_query']
                hard_match_nodes = preprocess_result['hard_match_nodes']
                query_intent = preprocess_result['intent']
                stream_gen = builder.stream_query_knowledge_graph(
                    enhanced_query, 
                    hard_match_nodes=hard_match_nodes,
                    query_intent=query_intent
                )
            else:
                stream_gen = builder.stream_query_knowledge_graph(msg)
            for item in stream_gen:
                if isinstance(item, str):
                    if item.startswith("错误:") or item.startswith("查询出错:"):
                         yield sse_event(create_error_event("search", item))
                    else:
                         yield sse_event({
                             "event": "delta",
                             "data": {"text": item}
                         })
                elif isinstance(item, dict) and item.get("type") == "graph_paths":
                    yield sse_event({
                        "event": "graph_data",
                        "data": item["data"]
                    })
                elif isinstance(item, dict) and item.get("type") == "retrieved_contexts":
                    yield sse_event({
                        "event": "contexts",
                        "data": item["data"]
                    })
                elif isinstance(item, dict) and item.get("type") == "done":
                    yield sse_event({
                        "event": "done",
                        "data": {
                            "full_answer": item.get("full_answer", ""),
                            "contexts": item.get("contexts", [])
                        }
                    })
                elif isinstance(item, dict):
                    yield sse_event(item)
                else:
                    logger.warning(f"未知的流式返回类型: {type(item)}")
        
        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',
                'Access-Control-Allow-Origin': '*'
            }
        )
    except Exception as e:
        logger.error(f"搜索接口出错: {e}")
        return Response(sse_event(create_error_event("unknown", str(e))), mimetype='text/event-stream'), 500

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
    logger.info("🔍 搜索接口: GET/POST /search")
    logger.info("🧬 生成向量: POST /nodes/generate_embeddings")
    
    # 启动Flask应用
    app.run(host='0.0.0.0', port=8001, debug=False, threaded=True)
