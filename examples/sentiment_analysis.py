"""
Advanced Sentiment Analysis with Multiple Models

This module compares different models for sentiment analysis.
"""

from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class SentimentAnalyzer:
    """Multi-model sentiment analyzer."""
    
    def __init__(self):
        self.models = {
            'distilbert': pipeline("sentiment-analysis", 
                                 model="distilbert-base-uncased-finetuned-sst-2-english"),
            'roberta': pipeline("sentiment-analysis", 
                              model="cardiffnlp/twitter-roberta-base-sentiment-latest"),
            'bert': pipeline("sentiment-analysis", 
                           model="nlptown/bert-base-multilingual-uncased-sentiment")
        }
    
    def analyze_text(self, text):
        """Analyze sentiment with all models."""
        results = {}
        for name, model in self.models.items():
            try:
                result = model(text)[0]
                results[name] = {
                    'label': result['label'],
                    'score': result['score']
                }
            except Exception as e:
                print(f"Error with {name}: {e}")
                results[name] = {'label': 'ERROR', 'score': 0.0}
        
        return results
    
    def analyze_batch(self, texts):
        """Analyze multiple texts."""
        results = []
        for text in texts:
            result = {'text': text}
            sentiment_results = self.analyze_text(text)
            result.update(sentiment_results)
            results.append(result)
        
        return pd.DataFrame(results)

def create_sample_reviews():
    """Create sample product reviews."""
    return [
        "This product exceeded my expectations! Amazing quality.",
        "Worst purchase ever. Complete waste of money.",
        "It's okay, nothing special but does the job.",
        "Absolutely love it! Will definitely buy again.",
        "Poor quality, broke after one week.",
        "Good value for money, happy with purchase.",
        "Terrible customer service, would not recommend.",
        "Perfect! Exactly what I was looking for."
    ]

def visualize_results(df):
    """Create visualization of sentiment analysis results."""
    plt.figure(figsize=(12, 8))
    
    # Convert labels to numeric for better visualization
    label_map = {'POSITIVE': 1, 'NEGATIVE': -1, 'NEUTRAL': 0}
    
    model_names = ['distilbert', 'roberta', 'bert']
    for i, model in enumerate(model_names):
        plt.subplot(2, 2, i+1)
        
        # Get scores and labels
        scores = []
        labels = []
        for _, row in df.iterrows():
            if model in row and isinstance(row[model], dict):
                score = row[model]['score']
                label = row[model]['label']
                
                # Map labels to sentiment
                if 'POS' in label.upper() or label.upper() == 'POSITIVE':
                    labels.append('Positive')
                    scores.append(score)
                elif 'NEG' in label.upper() or label.upper() == 'NEGATIVE':
                    labels.append('Negative')
                    scores.append(-score)  # Negative for visualization
                else:
                    labels.append('Neutral')
                    scores.append(0)
        
        # Create bar plot
        colors = ['green' if s > 0 else 'red' if s < 0 else 'gray' for s in scores]
        plt.bar(range(len(scores)), scores, color=colors, alpha=0.7)
        plt.title(f'{model.capitalize()} Sentiment Scores')
        plt.xlabel('Reviews')
        plt.ylabel('Sentiment Score')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """Run sentiment analysis comparison."""
    print("😊 Advanced Sentiment Analysis")
    print("-" * 40)
    
    # Create analyzer
    analyzer = SentimentAnalyzer()
    
    # Get sample reviews
    reviews = create_sample_reviews()
    
    # Analyze sentiments
    print("Analyzing sentiments...")
    results_df = analyzer.analyze_batch(reviews)
    
    # Display results
    print("\nResults:")
    for _, row in results_df.iterrows():
        print(f"\nText: {row['text']}")
        for model in ['distilbert', 'roberta', 'bert']:
            if model in row and isinstance(row[model], dict):
                result = row[model]
                print(f"  {model}: {result['label']} ({result['score']:.3f})")
    
    # Save results
    results_df.to_csv('sentiment_analysis_results.csv', index=False)
    print(f"\nResults saved to sentiment_analysis_results.csv")

if __name__ == "__main__":
    main() 