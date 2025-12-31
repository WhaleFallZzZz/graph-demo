
import unittest
import logging
from llama.utils import parse_llm_output_with_types

# Configure logging to show debug messages
logging.basicConfig(level=logging.DEBUG)

class TestParseLLMOutput(unittest.TestCase):

    def test_valid_json(self):
        """Test with valid JSON input"""
        json_str = '[{"head": "A", "head_type": "TypeA", "relation": "to", "tail": "B", "tail_type": "TypeB"}]'
        result = parse_llm_output_with_types(json_str)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['head'], "A")
        self.assertEqual(result[0]['head_type'], "TypeA")

    def test_invalid_json_simple_text(self):
        """Test with non-JSON text"""
        text = "This is not a JSON string."
        result = parse_llm_output_with_types(text)
        self.assertEqual(result, [])

    def test_invalid_json_regex_match(self):
        """
        Test with text that looks like triplets but is NOT valid JSON.
        Old behavior: Regex would extract this.
        New behavior: Should return empty list.
        """
        text = '"head": "A", "head_type": "TypeA", "relation": "to", "tail": "B", "tail_type": "TypeB"'
        result = parse_llm_output_with_types(text)
        self.assertEqual(result, [])

    def test_empty_string(self):
        """Test with empty string"""
        result = parse_llm_output_with_types("")
        self.assertEqual(result, [])

    def test_broken_json(self):
        """Test with broken JSON that cannot be fixed"""
        text = '[{"head": "A", "relation": "to"'  # Incomplete
        result = parse_llm_output_with_types(text)
        self.assertEqual(result, [])

if __name__ == '__main__':
    unittest.main()
