"""
Text Classification with Custom Data

This module shows how to fine-tune a model for custom text classification.
"""

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer
)
from datasets import Dataset
import torch
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np

class TextClassifier:
    """Custom text classifier using Hugging Face."""
    
    def __init__(self, model_name="distilbert-base-uncased", num_labels=2):
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        )
    
    def prepare_data(self, texts, labels=None):
        """Tokenize texts for model input."""
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        if labels is not None:
            encodings["labels"] = torch.tensor(labels)
        
        return encodings
    
    def predict(self, texts):
        """Make predictions on new texts."""
        self.model.eval()
        encodings = self.prepare_data(texts)
        
        with torch.no_grad():
            outputs = self.model(**encodings)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        return predictions.numpy()

def create_sample_dataset():
    """Create a sample dataset for demonstration."""
    data = {
        'text': [
            "I love this product! It's amazing.",
            "This is the worst thing I've ever bought.",
            "Great quality and fast shipping.",
            "Terrible customer service experience.",
            "Highly recommend this to everyone.",
            "Don't waste your money on this.",
            "Perfect for my needs, very satisfied.",
            "Complete disappointment, very poor quality."
        ],
        'label': [1, 0, 1, 0, 1, 0, 1, 0]  # 1: Positive, 0: Negative
    }
    return pd.DataFrame(data)

def main():
    """Demo text classification."""
    print("📝 Text Classification Example")
    print("-" * 40)
    
    # Create sample data
    df = create_sample_dataset()
    print("Sample dataset:")
    print(df)
    
    # Initialize classifier
    classifier = TextClassifier()
    
    # Make predictions
    test_texts = [
        "This product is fantastic!",
        "I hate this item.",
        "Pretty good overall."
    ]
    
    predictions = classifier.predict(test_texts)
    
    print("\nPredictions:")
    for text, pred in zip(test_texts, predictions):
        sentiment = "Positive" if pred[1] > pred[0] else "Negative"
        confidence = max(pred)
        print(f"Text: {text}")
        print(f"Prediction: {sentiment} (confidence: {confidence:.3f})")
        print()

if __name__ == "__main__":
    main() 