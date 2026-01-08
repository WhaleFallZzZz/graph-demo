"""
Unit tests for common utilities package.

Tests for:
- Text processing utilities
- JSON parsing utilities
- File operation utilities
- Cache management utilities
"""

import unittest
import tempfile
import os
import json
import time
from pathlib import Path

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from llama.common.text_utils import (
    clean_text,
    sanitize_for_neo4j,
    normalize_whitespace,
    remove_special_chars,
    extract_code_blocks,
    remove_think_tags,
    truncate_text,
    split_into_chunks,
    extract_sentences,
    normalize_text,
    count_words,
    count_characters,
    is_empty_or_whitespace
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
    get_file_hash,
    detect_file_type,
    is_supported_file,
    read_file_content,
    write_file_content,
    get_file_size,
    is_file_size_valid,
    get_file_extension,
    get_file_name,
    file_exists,
    directory_exists
)

from llama.common.cache_utils import (
    LRUCache,
    TTLCache,
    PersistentCache,
    CacheManager,
    cached
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
        self.assertEqual(sanitize_for_neo4j("test's name"), "test\\'s name")
        self.assertEqual(sanitize_for_neo4j("test\"quote"), "test\\\"quote")
        self.assertEqual(sanitize_for_neo4j("test\nnewline"), "test\\nnewline")
        
        # Test truncation
        long_text = "a" * 2000
        result = sanitize_for_neo4j(long_text, max_length=100)
        self.assertEqual(len(result), 100)
    
    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        self.assertEqual(normalize_whitespace("test  multiple   spaces"), "test multiple spaces")
        self.assertEqual(normalize_whitespace("  test  "), "test")
        self.assertEqual(normalize_whitespace(""), "")
    
    def test_remove_special_chars(self):
        """Test special character removal."""
        self.assertEqual(remove_special_chars("test@123!"), "test123")
        self.assertEqual(remove_special_chars("test(123)", keep_chars='()'), "test(123)")
        self.assertEqual(remove_special_chars(""), "")
    
    def test_extract_code_blocks(self):
        """Test code block extraction."""
        text = "```python\nprint('hello')\n```"
        blocks = extract_code_blocks(text)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].strip(), "print('hello')")
        
        # Test language filter
        blocks = extract_code_blocks(text, languages=['cypher'])
        self.assertEqual(len(blocks), 1)  # Should still extract, just not filter by language
    
    def test_remove_think_tags(self):
        """Test think tag removal."""
        text = "test </think>reasoning\nresult"
        result = remove_think_tags(text)
        self.assertEqual(result, "test result")
        
        # Test unclosed tags
        text = "test  unclosed"
        result = remove_think_tags(text)
        self.assertEqual(result, "test")  # Should remove unclosed tag and everything after
    
    def test_truncate_text(self):
        """Test text truncation."""
        self.assertEqual(truncate_text("This is a long text", 10), "This is...")
        self.assertEqual(truncate_text("short", 10), "short")
        self.assertEqual(truncate_text("", 10), "")
    
    def test_split_into_chunks(self):
        """Test text chunking."""
        text = "abcdefghij"
        chunks = split_into_chunks(text, 4, 2)
        self.assertEqual(len(chunks), 5)
        self.assertEqual(chunks, ['abcd', 'cdef', 'efgh', 'ghij', 'ij'])
    
    def test_extract_sentences(self):
        """Test sentence extraction."""
        text = "Hello world. How are you?"
        sentences = extract_sentences(text)
        self.assertEqual(len(sentences), 2)
        self.assertEqual(sentences[0], "Hello world")
        self.assertEqual(sentences[1], "How are you")
    
    def test_normalize_text(self):
        """Test text normalization."""
        self.assertEqual(normalize_text("Test123!", lowercase=True, remove_punct=True), "test123")
        self.assertEqual(normalize_text("Test123", remove_digits=True), "Test")
    
    def test_count_words(self):
        """Test word counting."""
        self.assertEqual(count_words("Hello world"), 2)
        self.assertEqual(count_words(""), 0)
        self.assertEqual(count_words("  test  "), 1)
    
    def test_count_characters(self):
        """Test character counting."""
        self.assertEqual(count_characters("Hello world"), 11)
        self.assertEqual(count_characters("Hello world", include_spaces=False), 10)
        self.assertEqual(count_characters(""), 0)
    
    def test_is_empty_or_whitespace(self):
        """Test empty/whitespace detection."""
        self.assertTrue(is_empty_or_whitespace(""))
        self.assertTrue(is_empty_or_whitespace("   "))
        self.assertFalse(is_empty_or_whitespace("test"))
        self.assertTrue(is_empty_or_whitespace(None))


class TestJsonUtils(unittest.TestCase):
    """Test cases for JSON utilities."""
    
    def test_safe_json_parse(self):
        """Test safe JSON parsing."""
        self.assertEqual(safe_json_parse('{"key": "value"}'), {"key": "value"})
        self.assertIsNone(safe_json_parse('invalid'))
        self.assertEqual(safe_json_parse('invalid', default={}), {})
        self.assertIsNone(safe_json_parse(""))
    
    def test_parse_llm_output(self):
        """Test LLM output parsing."""
        output = '[{"name": "test", "type": "disease"}]'
        result = parse_llm_output(output)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['name'], 'test')
        self.assertEqual(result[0]['type'], 'disease')
        
        # Test empty output
        self.assertEqual(parse_llm_output(""), [])
    
    def test_fix_json_syntax(self):
        """Test JSON syntax fixing."""
        self.assertEqual(fix_json_syntax("{'key': 'value',}"), '{"key": "value"}')
        self.assertEqual(fix_json_syntax('{"key": "value"}'), '{"key": "value"}')
    
    def test_extract_json_from_text(self):
        """Test JSON extraction from text."""
        text = 'Here is data: {"key": "value"}'
        result = extract_json_from_text(text)
        self.assertEqual(result, '{"key": "value"}')
        
        # Test code block
        text = '```json\n{"key": "value"}\n```'
        result = extract_json_from_text(text)
        self.assertEqual(result, '{"key": "value"}')
    
    def test_validate_json_structure(self):
        """Test JSON structure validation."""
        data = {'name': 'test', 'type': 'disease'}
        self.assertTrue(validate_json_structure(data, required_keys=['name']))
        self.assertFalse(validate_json_structure(data, required_keys=['missing']))
        self.assertTrue(validate_json_structure(data, expected_type=dict))
        self.assertFalse(validate_json_structure(data, expected_type=list))
    
    def test_json_to_csv(self):
        """Test JSON to CSV conversion."""
        data = [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]
        csv = json_to_csv(data)
        lines = csv.split('\n')
        self.assertEqual(len(lines), 3)
        self.assertIn('a,b', lines[0])
    
    def test_csv_to_json(self):
        """Test CSV to JSON conversion."""
        csv = 'a,b\n1,2\n3,4'
        data = csv_to_json(csv)
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0], {'a': '1', 'b': '2'})


class TestFileUtils(unittest.TestCase):
    """Test cases for file utilities."""
    
    def setUp(self):
        """Create temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_get_file_hash(self):
        """Test file hashing."""
        test_file = os.path.join(self.temp_dir, 'test.txt')
        write_file_content(test_file, 'test content')
        
        hash1 = get_file_hash(test_file)
        hash2 = get_file_hash(test_file)
        
        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 32)  # MD5 hash length
    
    def test_detect_file_type(self):
        """Test file type detection."""
        self.assertEqual(detect_file_type('document.pdf'), 'pdf')
        self.assertEqual(detect_file_type('document.txt'), 'text')
        self.assertEqual(detect_file_type('document.unknown'), None)
    
    def test_is_supported_file(self):
        """Test supported file check."""
        self.assertTrue(is_supported_file('document.pdf'))
        self.assertTrue(is_supported_file('document.txt'))
        self.assertFalse(is_supported_file('image.jpg'))
    
    def test_read_write_file(self):
        """Test file reading and writing."""
        test_file = os.path.join(self.temp_dir, 'test.txt')
        content = 'Hello, World!'
        
        write_file_content(test_file, content)
        self.assertTrue(file_exists(test_file))
        
        read_content = read_file_content(test_file)
        self.assertEqual(read_content, content)
    
    def test_get_file_size(self):
        """Test file size retrieval."""
        test_file = os.path.join(self.temp_dir, 'test.txt')
        write_file_content(test_file, 'test content')
        
        size = get_file_size(test_file)
        self.assertEqual(size, 12)  # "test content" length
    
    def test_get_file_extension(self):
        """Test file extension extraction."""
        self.assertEqual(get_file_extension('document.pdf'), 'pdf')
        self.assertEqual(get_file_extension('document'), '')
    
    def test_get_file_name(self):
        """Test file name extraction."""
        self.assertEqual(get_file_name('/path/to/document.pdf'), 'document.pdf')
        self.assertEqual(get_file_name('/path/to/document.pdf', with_extension=False), 'document')
    
    def test_file_directory_exists(self):
        """Test file/directory existence checks."""
        test_file = os.path.join(self.temp_dir, 'test.txt')
        write_file_content(test_file, 'test')
        
        self.assertTrue(file_exists(test_file))
        self.assertTrue(directory_exists(self.temp_dir))
        self.assertFalse(file_exists('/nonexistent/file.txt'))


class TestCacheUtils(unittest.TestCase):
    """Test cases for cache utilities."""
    
    def setUp(self):
        """Create temporary directory for cache tests."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_lru_cache(self):
        """Test LRU cache."""
        cache = LRUCache(capacity=3)
        
        cache.put('key1', 'value1')
        cache.put('key2', 'value2')
        cache.put('key3', 'value3')
        
        self.assertEqual(cache.get('key1'), 'value1')
        self.assertEqual(cache.get('key2'), 'value2')
        
        # Test eviction - key3 should be evicted (least recently used)
        cache.put('key4', 'value4')
        self.assertEqual(cache.get('key1'), 'value1')  # Still present
        self.assertEqual(cache.get('key2'), 'value2')  # Still present
        self.assertIsNone(cache.get('key3'))  # Should be evicted
        self.assertEqual(cache.get('key4'), 'value4')  # New item
        
        # Test stats
        stats = cache.get_stats()
        self.assertEqual(stats['size'], 3)
        self.assertEqual(stats['capacity'], 3)
        self.assertEqual(stats['evictions'], 1)
    
    def test_ttl_cache(self):
        """Test TTL cache."""
        cache = TTLCache(ttl_seconds=1)
        
        cache.put('key1', 'value1')
        self.assertEqual(cache.get('key1'), 'value1')
        
        # Wait for expiration
        time.sleep(1.5)
        self.assertIsNone(cache.get('key1'))
        
        # Test stats
        stats = cache.get_stats()
        self.assertEqual(stats['expirations'], 1)
    
    def test_persistent_cache(self):
        """Test persistent cache."""
        cache = PersistentCache(cache_dir=self.temp_dir, use_pickle=False)
        
        cache.put('key1', 'value1')
        self.assertEqual(cache.get('key1'), 'value1')
        
        # Test persistence
        cache2 = PersistentCache(cache_dir=self.temp_dir, use_pickle=False)
        self.assertEqual(cache2.get('key1'), 'value1')
        
        # Test size
        self.assertEqual(cache.size(), 1)
    
    def test_cache_manager(self):
        """Test unified cache manager."""
        cache = CacheManager(
            lru_capacity=100,
            ttl_seconds=3600,
            enable_persistent=False
        )
        
        cache.put('key1', 'value1')
        result = cache.get('key1')
        self.assertEqual(result, 'value1')
        
        # Test stats
        stats = cache.get_stats()
        self.assertIn('lru_cache', stats)
        self.assertIn('ttl_cache', stats)
    
    def test_cached_decorator(self):
        """Test cached decorator."""
        call_count = [0]
        
        @cached(ttl=3600)
        def expensive_function(x):
            call_count[0] += 1
            return x * 2
        
        # First call
        result1 = expensive_function(5)
        self.assertEqual(result1, 10)
        self.assertEqual(call_count[0], 1)
        
        # Second call (should use cache)
        result2 = expensive_function(5)
        self.assertEqual(result2, 10)
        self.assertEqual(call_count[0], 1)  # Should not increment


def run_tests():
    """Run all tests and return results."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestTextUtils))
    suite.addTests(loader.loadTestsFromTestCase(TestJsonUtils))
    suite.addTests(loader.loadTestsFromTestCase(TestFileUtils))
    suite.addTests(loader.loadTestsFromTestCase(TestCacheUtils))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    result = run_tests()
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
