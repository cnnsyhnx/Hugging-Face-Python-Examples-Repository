"""
Text preprocessing utilities for Hugging Face examples.
"""

import pandas as pd
import re
from typing import List, Dict, Any

def clean_text(text: str) -> str:
    """Clean and preprocess text."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove special characters (optional)
    # text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    return text

def preprocess_text(texts: List[str]) -> List[str]:
    """Preprocess a list of texts."""
    return [clean_text(text) for text in texts]

def load_data(file_path: str, text_column: str = 'text', label_column: str = 'label') -> Dict[str, Any]:
    """Load data from various formats."""
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        df = pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file format")
    
    return {
        'texts': df[text_column].tolist(),
        'labels': df[label_column].tolist() if label_column in df.columns else None,
        'dataframe': df
    }

def create_train_val_split(data: Dict[str, Any], val_ratio: float = 0.2) -> Dict[str, Any]:
    """Split data into training and validation sets."""
    from sklearn.model_selection import train_test_split
    
    if data['labels'] is not None:
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            data['texts'], data['labels'], test_size=val_ratio, random_state=42
        )
        return {
            'train': {'texts': train_texts, 'labels': train_labels},
            'val': {'texts': val_texts, 'labels': val_labels}
        }
    else:
        train_texts, val_texts = train_test_split(
            data['texts'], test_size=val_ratio, random_state=42
        )
        return {
            'train': {'texts': train_texts},
            'val': {'texts': val_texts}
        } 