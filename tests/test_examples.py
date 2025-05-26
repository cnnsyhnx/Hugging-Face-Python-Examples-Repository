"""
Basic tests for the examples.
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.basic_pipelines import sentiment_analysis_example
from utils.data_helpers import clean_text, preprocess_text

class TestExamples(unittest.TestCase):
    """Test basic functionality."""
    
    def test_clean_text(self):
        """Test text cleaning function."""
        text = "  Hello    world!  "
        cleaned = clean_text(text)
        self.assertEqual(cleaned, "Hello world!")
    
    def test_preprocess_text(self):
        """Test text preprocessing."""
        texts = ["  Hello  ", "  World  "]
        processed = preprocess_text(texts)
        self.assertEqual(processed, ["Hello", "World"])
    
    def test_sentiment_analysis(self):
        """Test sentiment analysis pipeline."""
        # This is a basic test - in practice you'd mock the pipeline
        try:
            sentiment_analysis_example()
            self.assertTrue(True)  # If no exception, test passes
        except Exception as e:
            self.fail(f"Sentiment analysis failed: {e}")

if __name__ == '__main__':
    unittest.main() 