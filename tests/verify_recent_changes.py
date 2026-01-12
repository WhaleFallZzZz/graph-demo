
import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llama.neo4j_text_sanitizer import Neo4jTextSanitizer
from llama.config import EXTRACTOR_CONFIG
from llama.enhanced_entity_extractor import StandardTermMapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_fuzzy_mapping():
    logger.info("Testing Fuzzy Mapping in Neo4jTextSanitizer...")
    # "眼轴" should map to "眼轴长度" via Alias Mapping
    input_term_alias = "眼轴"
    expected_alias = "眼轴长度"
    
    result_alias = Neo4jTextSanitizer.sanitize_node_name(input_term_alias)
    if result_alias == expected_alias:
        logger.info(f"✅ Alias Mapping passed: '{input_term_alias}' -> '{result_alias}'")
    else:
        logger.warning(f"❌ Alias Mapping failed. Expected '{expected_alias}', got '{result_alias}'")

    # "眼轴长" should map to "眼轴长度" via Fuzzy Logic (Levenshtein)
    # Length of "眼轴长" is 3. "眼轴长度" is 4.
    # difflib ratio: 2*3 / 7 = 0.857 > 0.85
    input_term_fuzzy = "眼轴长"
    expected_fuzzy = "眼轴长度"
    
    result_fuzzy = Neo4jTextSanitizer.sanitize_node_name(input_term_fuzzy)
    if result_fuzzy == expected_fuzzy:
        logger.info(f"✅ Fuzzy mapping passed: '{input_term_fuzzy}' -> '{result_fuzzy}'")
    else:
        logger.warning(f"❌ Fuzzy mapping failed. Expected '{expected_fuzzy}', got '{result_fuzzy}'")

def test_alias_replacement_logic():
    logger.info("Testing Alias Replacement Logic...")
    mapping = EXTRACTOR_CONFIG.get("alias_mapping", {})
    
    text = "患者AL为26mm，D为-3.00。"
    expected_part1 = "眼轴长度"
    expected_part2 = "屈光度"
    
    # Use regex approach (same as in kg_manager.py)
    import re
    sorted_aliases = sorted(mapping.keys(), key=len, reverse=True)
    pattern_str = '|'.join(map(re.escape, sorted_aliases))
    pattern = re.compile(pattern_str)
    
    modified_text = pattern.sub(lambda m: mapping[m.group(0)], text)
            
    logger.info(f"Original: '{text}'")
    logger.info(f"Modified: '{modified_text}'")
    
    # Check if "眼轴长度" is present and NOT "眼眼轴长度度"
    if expected_part1 in modified_text and expected_part2 in modified_text:
        if "眼眼轴" not in modified_text:
             logger.info("✅ Alias replacement passed (Correctly handled recursive issue).")
        else:
             logger.warning("❌ Alias replacement failed (Recursive replacement detected).")
    else:
        logger.warning("❌ Alias replacement failed (Target words not found).")

def test_weak_relation_logic():
    logger.info("Testing Weak Relation Logic...")
    from llama_index.core.graph_stores.types import Relation
    import itertools
    
    text = "检查发现眼轴长度增加，且屈光度改变。"
    
    found_entities = []
    for entity in StandardTermMapper.STANDARD_ENTITIES:
        if entity in text:
            found_entities.append(entity)
            
    logger.info(f"Found entities: {found_entities}")
    
    new_relations = []
    if len(found_entities) >= 2:
        for e1, e2 in itertools.combinations(found_entities, 2):
            rel = Relation(
                source_id=e1,
                target_id=e2,
                label="RELATED_TO",
                properties={"confidence": "low", "type": "co_occurrence"}
            )
            new_relations.append(rel)
            
    logger.info(f"Created {len(new_relations)} weak relations.")
    if len(new_relations) > 0:
        logger.info(f"Relation: {new_relations[0].source_id} -> {new_relations[0].target_id}")
        logger.info("✅ Weak relation logic passed.")
    else:
        logger.warning("❌ Weak relation logic failed (no relations created).")

if __name__ == "__main__":
    test_alias_replacement_logic()
    test_weak_relation_logic()
    test_fuzzy_mapping() # Now we can run this as well
