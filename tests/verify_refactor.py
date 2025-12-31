
import sys
import os
import time
import logging
from pathlib import Path
from unittest.mock import MagicMock
from typing import Any, Optional

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'llama')))

from ocr_parser import DeepSeekOCRParser
from entity_extractor import MultiStageLLMExtractor
from llama_index.core.schema import TextNode
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleMockLLM(CustomLLM):
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        time.sleep(0.5) # Simulate LLM latency
        if "实体识别" in prompt:
            text = '[{"name": "近视", "type": "眼部疾病"}, {"name": "角膜塑形镜", "type": "视光产品"}]'
        else:
            text = '[{"head": "近视", "head_type": "眼部疾病", "relation": "治疗", "tail": "角膜塑形镜", "tail_type": "视光产品"}]'
        return CompletionResponse(text=text)

    def stream_complete(self, prompt: str, **kwargs: Any):
        pass

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata()

def test_ocr_multithreading():
    logger.info("Testing OCR Multi-threading...")
    
    # Check CPU count
    cpu_count = os.cpu_count() or 1
    logger.info(f"Detected CPU count: {cpu_count}")
    
    # Find a PDF
    pdf_path = Path("角膜塑形镜验配技术  基础篇.pdf")
    if pdf_path.exists():
        pdf_file = pdf_path
    else:
        pdf_files = list(Path("wait_build").glob("*.pdf"))
        if not pdf_files:
            logger.warning("No PDF files found. Skipping OCR test.")
            return
        pdf_file = pdf_files[0]
    
    logger.info(f"Using PDF: {pdf_file}")
    
    # Initialize parser with limit and dummy key to bypass init check
    parser = DeepSeekOCRParser(api_key="dummy_key", max_pages=3)
    
    # Mock the internal API call
    original_create = parser.client.chat.completions.create
    
    def mock_create(*args, **kwargs):
        time.sleep(1) # Simulate network delay
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Page content simulation."
        return mock_response
        
    parser.client.chat.completions.create = mock_create
    
    start_time = time.time()
    docs = parser.load_data(pdf_file)
    end_time = time.time()
    
    duration = end_time - start_time
    logger.info(f"OCR Processing took {duration:.2f} seconds")
    
    # Threshold logic: 
    # If cpu_count >= 3, expected time is roughly 1s + overhead. 
    # If cpu_count < 3, it might take longer.
    # We add ample buffer for overhead (rendering pages takes time).
    # Let's say overhead per page is 0.5s.
    
    if cpu_count >= 3:
        threshold = 3.5 # Relaxed threshold
    else:
        threshold = 5.0
        
    if duration < threshold:
        logger.info(f"✅ OCR Multi-threading verified (time < {threshold}s for 3 pages)")
    else:
        logger.warning(f"❌ OCR Multi-threading might not be working efficiently (time {duration:.2f}s >= {threshold}s)")

def test_llm_multistage():
    logger.info("Testing LLM Multi-stage Extraction...")
    
    mock_llm = SimpleMockLLM()
    
    # Mock Graph Store
    mock_graph_store = MagicMock()
    
    extractor = MultiStageLLMExtractor(
        llm=mock_llm,
        entity_prompt="实体识别 prompt {text}",
        relation_prompt="关系抽取 prompt {text} {entities}",
        num_workers=4,
        graph_store=mock_graph_store
    )
    
    nodes = [TextNode(text=f"Node {i} text", metadata={"file_name": "test_doc.pdf"}) for i in range(4)]
    
    # Remove previous test output if exists
    # if os.path.exists("test_doc.pdf-json"):
    #     os.remove("test_doc.pdf-json")
    
    # Clean up test output dir
    import shutil
    import datetime
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    output_dir = os.path.join(os.getcwd(), "llm_outputs", today_str)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    start_time = time.time()
    results = extractor.extract(nodes)
    end_time = time.time()
    
    duration = end_time - start_time
    logger.info(f"LLM Extraction took {duration:.2f} seconds")
    
    # 4 nodes. 
    # Stage 1: 4 calls (parallel). Max time ~0.5s.
    # Stage 2: 4 calls (consumer). 
    # If 2 consumers: 2 rounds of 0.5s = 1s.
    # Total ~1.5s.
    
    if duration < 3.0:
        logger.info("✅ LLM Multi-stage verified (time < 3s for 4 nodes)")
    else:
        logger.warning("❌ LLM Multi-stage might not be working efficiently")
        
    # Check results
    assert len(results) == 4
    # Check if graph_store was called
    assert mock_graph_store.upsert_nodes.call_count > 0
    assert mock_graph_store.upsert_relations.call_count > 0
    
    # Check result structure - should indicate saved_to_neo4j
    assert "saved_to_neo4j" in results[0] or "kg_triplets" in results[0]
    if "saved_to_neo4j" in results[0]:
        logger.info("✅ Direct Neo4j streaming verified")
    else:
        logger.warning("❌ Direct Neo4j streaming NOT verified (fallback used?)")

    # Verify JSON file creation in llm_outputs/{date}
    import datetime
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    output_dir = os.path.join(os.getcwd(), "llm_outputs", today_str)
    output_file = os.path.join(output_dir, "test_doc-json.txt")
    
    if os.path.exists(output_file):
        logger.info(f"✅ JSON output file created: {output_file}")
        # Check content
        with open(output_file, "r", encoding="utf-8") as f:
            content = f.read()
            # Verify basic structure
            if "/* 处理时间:" in content and '"node_id":' in content:
                logger.info("✅ JSON output file content format verified")
            else:
                logger.warning("❌ JSON output file content format incorrect")
            
            # Count JSON blocks roughly
            block_count = content.count('"node_id":')
            if block_count == 4:
                logger.info(f"✅ JSON output file contains correct number of blocks: {block_count}")
            else:
                logger.warning(f"❌ JSON output file contains {block_count} blocks, expected 4")
        
        # Cleanup
        # os.remove(output_file) # Keep for inspection or remove
    else:
        logger.error(f"❌ JSON output file NOT created at {output_file}")

    logger.info("✅ Result structure verified")

if __name__ == "__main__":
    test_ocr_multithreading()
    test_llm_multistage()
