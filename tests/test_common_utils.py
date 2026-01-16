"""
Unit tests for common utilities package.

Tests for:
- Text processing utilities
- JSON parsing utilities
- File operation utilities
"""

import unittest
import tempfile
import os
import json
from pathlib import Path

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from llama.common.text_utils import (
    clean_text,
    sanitize_for_neo4j
)

from llama.common.json_utils import (
    safe_json_parse,
    parse_llm_output,
    fix_json_syntax,
    extract_json_from_text,
    validate_json_structure,
    json_to_csv,
    csv_to_json
)

from llama.common.file_utils import (
    get_file_hash
)


class TestTextUtils(unittest.TestCase):
    """Test cases for text utilities."""
    
    def test_clean_text(self):
        """Test text cleaning."""
        self.assertEqual(clean_text("_test_text_"), "testtext")
        self.assertEqual(clean_text("  test  "), "test")
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text(None), None)
    
    def test_sanitize_for_neo4j(self):
        """Test Neo4j text sanitization."""
        result = sanitize_for_neo4j("test's name")
        self.assertEqual(result, "test\\'s name")
        self.assertEqual(repr(result), "\"test\\\\'s name\"")
        
        result = sanitize_for_neo4j("test\"quote")
        self.assertEqual(result, "test\\\"quote")
        self.assertEqual(repr(result), "'test\\\\\"quote'")
        
        result = sanitize_for_neo4j("test\nnewline")
        self.assertEqual(result, "test\\nnewline")
        self.assertEqual(repr(result), "'test\\\\nnewline'")
        
        # Test truncation
        long_text = "a" * 2000
        result = sanitize_for_neo4j(long_text, max_length=100)
        self.assertEqual(len(result), 100)


class TestJsonUtils(unittest.TestCase):
    """Test cases for JSON utilities."""
    
    def test_safe_json_parse(self):
        """Test safe JSON parsing."""
        result = safe_json_parse('{"key": "value"}')
        self.assertEqual(result, {"key": "value"})
        
        # Test invalid JSON
        result = safe_json_parse('invalid')
        self.assertIsNone(result)
    
    def test_parse_llm_output(self):
        """Test LLM output parsing."""
        # Test JSON array format
        output = '[{"name": "test", "type": "disease"}]'
        result = parse_llm_output(output)
        self.assertEqual(result, [{'name': 'test', 'type': 'disease'}])
        
        # Test JSON object with entities key
        output = '{"entities": [{"name": "test", "type": "disease"}]}'
        result = parse_llm_output(output)
        self.assertEqual(result, [{'name': 'test', 'type': 'disease'}])
        
        # Test single JSON object
        output = '{"name": "test", "type": "disease"}'
        result = parse_llm_output(output)
        self.assertEqual(result, [{'name': 'test', 'type': 'disease'}])
    
    def test_fix_json_syntax(self):
        """Test JSON syntax fixing."""
        broken = "{'key': 'value'}"
        fixed = fix_json_syntax(broken)
        self.assertEqual(fixed, '{"key": "value"}')
    
    def test_extract_json_from_text(self):
        """Test JSON extraction from text."""
        text = 'Some text {"key": "value"} more text'
        result = extract_json_from_text(text)
        self.assertEqual(result, '{"key": "value"}')
    
    def test_validate_json_structure(self):
        """Test JSON structure validation."""
        valid = {"head": "test", "relation": "test", "tail": "test"}
        self.assertTrue(validate_json_structure(valid))
        
        invalid = {"head": "test"}
        self.assertFalse(validate_json_structure(invalid))
    
    def test_json_to_csv(self):
        """Test JSON to CSV conversion."""
        data = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
        csv = json_to_csv(data)
        self.assertIn("name,age", csv)
        self.assertIn("Alice,30", csv)
    
    def test_csv_to_json(self):
        """Test CSV to JSON conversion."""
        csv = "name,age\nAlice,30\nBob,25"
        result = csv_to_json(csv)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["name"], "Alice")


class TestFileUtils(unittest.TestCase):
    """Test cases for file utilities."""
    
    def setUp(self):
        """Create temporary directory for file tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test.txt")
    
    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_get_file_hash(self):
        """Test file hash calculation."""
        with open(self.test_file, 'w') as f:
            f.write("test content")
        
        hash1 = get_file_hash(self.test_file)
        hash2 = get_file_hash(self.test_file)
        self.assertEqual(hash1, hash2)


def run_tests():
    """Run all tests and return results."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestTextUtils))
    suite.addTests(loader.loadTestsFromTestCase(TestJsonUtils))
    suite.addTests(loader.loadTestsFromTestCase(TestFileUtils))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    result = run_tests()
    sys.exit(0 if result.wasSuccessful() else 1)
