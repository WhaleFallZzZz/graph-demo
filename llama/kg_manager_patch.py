    def stream_query_knowledge_graph(self, query: str, index: Any = None) -> Any:
        """
        流式查询知识图谱，返回生成器
        
        Args:
            query: 查询字符串
            index: 图谱索引（可选）
            
        Returns:
            生成器，依次生成：
            1. LLM回答的文本片段 (str)
            2. 最终的图谱路径数据 (dict)
        """
        try:
            logger.info(f"开始流式查询: {query}")
            
            if index is None:
                if not self.graph_store:
                    yield "错误: 图存储未初始化"
                    return
                
                # 确保LLM和Embed Model已就绪
                if not self.llm or not self.embed_model:
                     if not self.initialize():
                         yield "错误: 组件初始化失败"
                         return
                
                try:
                    index = self.modules['PropertyGraphIndex'].from_existing(
                        property_graph_store=self.graph_store,
                        llm=self.llm,
                        embed_model=self.embed_model
                    )
                except Exception as e:
                    logger.error(f"加载现有索引失败: {e}")
                    yield f"加载索引失败: {str(e)}"
                    return
            
            # 创建流式查询引擎
            query_engine = index.as_query_engine(
                include_text=True,
                similarity_top_k=5,
                streaming=True  # 启用流式输出
            )
            
            # 添加后处理器 (与同步版本保持一致)
            postprocessors = []
            initial_k = 5
            
            try:
                from semantic_enrichment_postprocessor import SemanticEnrichmentPostprocessor
                semantic_enricher = SemanticEnrichmentPostprocessor(
                    graph_store=self.graph_store,
                    max_neighbors_per_entity=10
                )
                postprocessors.append(semantic_enricher)
            except Exception as e:
                logger.warning(f"语义补偿后处理器初始化失败: {e}")
            
            reranker = RerankerFactory.create_reranker()
            if reranker:
                initial_k = RERANK_CONFIG.get('initial_top_k', 10)
                postprocessors.append(reranker)
                
            if postprocessors:
                query_engine = index.as_query_engine(
                    include_text=True,
                    similarity_top_k=initial_k,
                    node_postprocessors=postprocessors,
                    streaming=True
                )
            
            # 执行查询，获取流式响应对象
            streaming_response = query_engine.query(query)
            
            # 1. 实时推送LLM生成的文本
            full_answer = ""
            for token in streaming_response.response_gen:
                full_answer += token
                yield token
            
            logger.info("LLM文本生成完成，开始提取图谱路径...")
            
            # 2. 异步搜索路径 (基于生成的完整答案和源节点)
            # 构造一个模拟的response对象，包含source_nodes，用于路径提取
            # streaming_response 对象本身包含 source_nodes
            paths = self._extract_graph_paths(query, streaming_response, index)
            
            logger.info(f"路径提取完成，找到 {len(paths)} 条路径")
            
            # 3. 推送图谱路径数据 (作为最后一个特殊的yield)
            yield {
                "type": "graph_paths",
                "data": paths,
                "full_answer": full_answer # 可选：返回完整答案以供校验
            }
            
        except Exception as e:
            logger.error(f"流式查询失败: {e}")
            yield f"查询出错: {str(e)}"
