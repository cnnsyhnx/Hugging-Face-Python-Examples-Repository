"""
Custom Model Training Example

This module demonstrates how to fine-tune a pre-trained model for a custom task.
"""

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import Dict, List, Tuple

class CustomTrainer:
    """Custom model trainer using Hugging Face."""
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 2,
        learning_rate: float = 2e-5,
        batch_size: int = 16,
        num_epochs: int = 3
    ):
        """Initialize the trainer."""
        self.model_name = model_name
        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        
        # Set up training arguments
        self.training_args = TrainingArguments(
            output_dir="./results",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
    
    def prepare_dataset(self, texts: List[str], labels: List[int]) -> Dataset:
        """Prepare dataset for training."""
        # Tokenize texts
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512
        )
        
        # Create dataset
        dataset = Dataset.from_dict({
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': labels
        })
        
        return dataset
    
    def compute_metrics(self, pred) -> Dict:
        """Compute metrics for evaluation."""
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='weighted'
        )
        acc = accuracy_score(labels, preds)
        
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train(
        self,
        train_texts: List[str],
        train_labels: List[int],
        eval_texts: List[str] = None,
        eval_labels: List[int] = None
    ) -> Trainer:
        """Train the model."""
        # Prepare datasets
        train_dataset = self.prepare_dataset(train_texts, train_labels)
        
        if eval_texts and eval_labels:
            eval_dataset = self.prepare_dataset(eval_texts, eval_labels)
        else:
            eval_dataset = None
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            data_collator=DataCollatorWithPadding(self.tokenizer)
        )
        
        # Train model
        trainer.train()
        
        return trainer

def create_sample_data():
    """Create sample data for demonstration."""
    data = {
        'text': [
            "This product is amazing! I love it.",
            "Terrible quality, would not recommend.",
            "Great value for money.",
            "Poor customer service experience.",
            "Excellent product, highly recommended.",
            "Waste of money, very disappointed.",
            "Good quality and fast shipping.",
            "Awful experience, avoid at all costs."
        ],
        'label': [1, 0, 1, 0, 1, 0, 1, 0]  # 1: Positive, 0: Negative
    }
    return pd.DataFrame(data)

def main():
    """Run custom training example."""
    print("🎯 Custom Model Training Example")
    print("-" * 40)
    
    # Create sample data
    df = create_sample_data()
    print("Sample dataset:")
    print(df)
    
    # Split data
    from sklearn.model_selection import train_test_split
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Initialize trainer
    trainer = CustomTrainer(
        model_name="distilbert-base-uncased",
        num_labels=2,
        learning_rate=2e-5,
        batch_size=8,
        num_epochs=3
    )
    
    # Train model
    print("\nTraining model...")
    trainer.train(
        train_texts=train_df['text'].tolist(),
        train_labels=train_df['label'].tolist(),
        eval_texts=eval_df['text'].tolist(),
        eval_labels=eval_df['label'].tolist()
    )
    
    print("\nTraining complete! Model saved to ./results")

if __name__ == "__main__":
    main() 