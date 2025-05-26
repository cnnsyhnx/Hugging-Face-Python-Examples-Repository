"""
Basic Hugging Face Pipelines Examples

This module demonstrates the simplest way to use Hugging Face models
through the pipeline API.
"""

from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

def sentiment_analysis_example():
    """Simple sentiment analysis example."""
    print("🎯 Sentiment Analysis Example")
    print("-" * 40)
    
    # Initialize pipeline
    classifier = pipeline("sentiment-analysis")
    
    # Test sentences
    texts = [
        "I love using Hugging Face!",
        "This movie was terrible.",
        "The weather is okay today.",
        "Amazing work on this project!"
    ]
    
    # Analyze sentiment
    for text in texts:
        result = classifier(text)[0]
        print(f"Text: {text}")
        print(f"Result: {result['label']} (confidence: {result['score']:.3f})")
        print()

def text_generation_example():
    """Simple text generation example."""
    print("✍️ Text Generation Example")
    print("-" * 40)
    
    # Initialize pipeline
    generator = pipeline("text-generation", model="gpt2")
    
    # Generate text
    prompt = "The future of artificial intelligence"
    results = generator(
        prompt, 
        max_length=100, 
        num_return_sequences=2,
        temperature=0.7,
        pad_token_id=50256
    )
    
    print(f"Prompt: {prompt}")
    print("\nGenerated texts:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['generated_text']}")

def question_answering_example():
    """Simple question answering example."""
    print("❓ Question Answering Example")
    print("-" * 40)
    
    # Initialize pipeline
    qa_pipeline = pipeline("question-answering")
    
    # Context and questions
    context = """
    Hugging Face is a company that develops tools for building applications 
    using machine learning. They are famous for their Transformers library, 
    which provides pre-trained models for natural language processing tasks. 
    The company was founded in 2016 and is based in New York and Paris.
    """
    
    questions = [
        "What is Hugging Face famous for?",
        "When was the company founded?",
        "Where is Hugging Face based?"
    ]
    
    print("Context:", context.strip())
    print("\nQuestions and Answers:")
    
    for question in questions:
        result = qa_pipeline(question=question, context=context)
        print(f"\nQ: {question}")
        print(f"A: {result['answer']} (confidence: {result['score']:.3f})")

def main():
    """Run all pipeline examples."""
    print("🤗 Hugging Face Pipeline Examples")
    print("=" * 50)
    
    sentiment_analysis_example()
    print("\n" + "="*50 + "\n")
    
    text_generation_example()
    print("\n" + "="*50 + "\n")
    
    question_answering_example()

if __name__ == "__main__":
    main() 