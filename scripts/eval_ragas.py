import os
import sys
import re
from pathlib import Path
from dotenv import load_dotenv
import httpx

# 加载环境变量
PROJECT_ROOT = Path(__file__).parent.parent
env_path = PROJECT_ROOT / '.env'
load_dotenv(env_path)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# ！！！在 Python 3.11 环境下，我们通常不需要复杂的 Patch ！！！
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from llama.config import API_CONFIG
from llama.kg_manager import builder
from typing import List, Any, Optional
from langchain_core.outputs import LLMResult, Generation
from langchain_core.prompt_values import PromptValue

def parse_test_file(file_path):
    """解析测试文件，提取问题、上下文和正确答案"""
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在")
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 使用正则表达式匹配 "数字. 问题：" 格式，这能匹配文件中的 1. 到 20.
    # 格式示例: 1. 问题：... \n 上下文：... \n 正确答案：...
    pattern = r'(\d+)\.\s*问题：(.*?)\n\s*上下文：(.*?)\n\s*正确答案：(.*?)(?=\n\d+\.|$)'
    matches = re.finditer(pattern, content, re.DOTALL)
    
    samples = []
    for match in matches:
        samples.append({
            "question": match.group(2).strip(),
            "context": match.group(3).strip(),
            "reference": match.group(4).strip()
        })
            
    print(f"成功从 {file_path} 解析出 {len(samples)} 个测试样本")
    return samples

class CustomChatOpenAI(ChatOpenAI):
    """
    自定义 ChatOpenAI，强制拦截并修改参数以适配不支持 n > 1 的模型（如 SiliconFlow 上的某些型号）
    """
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if "n" in kwargs and kwargs["n"] > 1:
            kwargs["n"] = 1
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        if "n" in kwargs and kwargs["n"] > 1:
            kwargs["n"] = 1
        return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)

def main():
    print("=" * 60)
    print("Ragas 自动化评估脚本 (集成 Search 接口)")
    print("=" * 60)
    
    # 1. 初始化知识图谱构建器（用于 Search）
    print("正在初始化知识图谱组件...")
    if not builder.initialize():
        print("错误: 知识图谱组件初始化失败")
        return

    # 2. 解析测试文件
    test_file = PROJECT_ROOT / "tests" / "青少年.txt"
    samples = parse_test_file(test_file)
    if not samples:
        print("未找到测试样本，退出")
        return

    # 3. 为每个问题获取模型答案
    print("\n正在通过 Search 接口获取模型回答...")
    user_inputs = []
    retrieved_contexts = []
    responses = []
    references = []
    
    for i, s in enumerate(samples):
        question = s['question']
        print(f"[{i+1}/{len(samples)}] 正在请求: {question}")
        
        try:
            # 调用 query_knowledge_graph（不是 stream_query_knowledge_graph）
            search_result = builder.query_knowledge_graph(question)
            
            # 确保 search_result 是字典类型
            if not isinstance(search_result, dict):
                print(f"  警告: 查询返回类型异常: {type(search_result)}")
                search_result = {"answer": "查询返回格式异常", "contexts": [], "paths": []}
            
            answer = search_result.get("answer", "回答失败")
            contexts = search_result.get("contexts", [])
            
            # 确保 contexts 是列表
            if not isinstance(contexts, list):
                contexts = [str(contexts)] if contexts else []
                
        except Exception as e:
            print(f"  查询失败: {e}")
            import traceback
            traceback.print_exc()
            answer = f"查询失败: {str(e)}"
            contexts = []
        
        user_inputs.append(question)
        # 使用检索到的真实上下文
        retrieved_contexts.append(contexts if contexts else ["未检索到相关上下文"]) 
        responses.append(answer)
        references.append(s['reference'])
        
        print(f"  回答: {answer[:50]}...")

    # 4. 初始化 Ragas 评估模型（用于打分）
    api_key = API_CONFIG["siliconflow"]["api_key"]
    siliconflow_base_url = "https://api.siliconflow.cn/v1"
    
    # 禁用 SSL 验证，确保连接稳定
    custom_client = httpx.Client(verify=False)
    
    # 使用自定义 CustomChatOpenAI 强制拦截 n > 1 
    lc_llm = CustomChatOpenAI(
        model_name="Qwen/Qwen3-VL-30B-A3B-Thinking",
        openai_api_key=api_key, 
        openai_api_base=siliconflow_base_url, 
        temperature=0.0,
        timeout=600.0,
        http_client=custom_client
    )
    # 使用官方 Wrapper 以确保内部接口对齐，通常官方 Wrapper 已经处理了 n=1 的问题
    ragas_llm = LangchainLLMWrapper(lc_llm)
    
    lc_embeddings = OpenAIEmbeddings(
        model="BAAI/bge-m3", 
        openai_api_key=api_key, 
        openai_api_base=siliconflow_base_url,
        timeout=120.0,
        http_client=custom_client
        # 注意: LangChain 会自动处理批量 embedding 请求，无需手动配置
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(lc_embeddings)

    # 5. 构建数据集并评估
    data = {
        "user_input": user_inputs,
        "retrieved_contexts": retrieved_contexts,
        "response": responses,
        "reference": references
    }
    dataset = Dataset.from_dict(data)

    print("\n开始评估 (使用 ragas.evaluate)...")
    try:
        # 提高并发以加快速度，由于 SiliconFlow RPM 较高，2-4 个线程是合理的
        import os
        os.environ["RAGAS_DO_NOT_TRACK"] = "true"
        max_workers = 4 
        print(f"使用 {max_workers} 个并行工作线程进行评估...")
        run_config = RunConfig(
            max_workers=max_workers, 
            timeout=300, 
            max_retries=3,
            max_wait=60,
            log_tenacity=True
        )
        result = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            run_config=run_config
        )
        
        print("\n" + "=" * 60)
        print("评估完成")
        print("=" * 60)
        print(result)
        
        # 导出详细结果进行分析
        df = result.to_pandas()
        
        # 调整列顺序，将 user_input 和 retrieved_contexts 放到最前面
        cols = df.columns.tolist()
        # 优先确保这几个核心列在前面
        main_cols = ['user_input', 'retrieved_contexts', 'response', 'reference']
        other_cols = [c for c in cols if c not in main_cols]
        df = df[main_cols + other_cols]
        
        output_csv = PROJECT_ROOT / "evaluation_results.csv"
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"\n详细得分已保存至: {output_csv}")
        
    except Exception as e:
        print(f"\n评估失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
