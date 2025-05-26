"""
Question Answering with Custom Context

This module demonstrates how to use Hugging Face models for question answering.
"""

from transformers import pipeline
import pandas as pd
from typing import List, Dict

class QuestionAnswerer:
    """Question answering system using Hugging Face."""
    
    def __init__(self, model_name: str = "deepset/roberta-base-squad2"):
        """Initialize the question answering pipeline."""
        self.qa_pipeline = pipeline(
            "question-answering",
            model=model_name,
            tokenizer=model_name
        )
    
    def answer_question(self, question: str, context: str) -> Dict:
        """Answer a single question given context."""
        result = self.qa_pipeline(
            question=question,
            context=context
        )
        return {
            'answer': result['answer'],
            'confidence': result['score'],
            'start': result['start'],
            'end': result['end']
        }
    
    def answer_questions(self, questions: List[str], context: str) -> List[Dict]:
        """Answer multiple questions given context."""
        results = []
        for question in questions:
            result = self.answer_question(question, context)
            results.append({
                'question': question,
                **result
            })
        return results

def create_sample_context():
    """Create a sample context for demonstration."""
    return """
    Artificial Intelligence (AI) is intelligence demonstrated by machines, as opposed to natural 
    intelligence displayed by animals including humans. AI research has been defined as the field 
    of study of intelligent agents, which refers to any system that perceives its environment and 
    takes actions that maximize its chance of achieving its goals. The term "artificial intelligence" 
    had previously been used to describe machines that mimic and display "human" cognitive skills 
    that are associated with the human mind, such as "learning" and "problem-solving". This definition 
    has since been rejected by major AI researchers who now describe AI in terms of rationality and 
    acting rationally, which does not limit how intelligence can be articulated.
    """

def create_sample_questions():
    """Create sample questions for demonstration."""
    return [
        "What is Artificial Intelligence?",
        "How is AI research defined?",
        "What skills were previously associated with AI?",
        "How do modern AI researchers describe AI?"
    ]

def main():
    """Run question answering example."""
    print("❓ Question Answering Example")
    print("-" * 40)
    
    # Create question answerer
    qa = QuestionAnswerer()
    
    # Get sample context and questions
    context = create_sample_context()
    questions = create_sample_questions()
    
    print("Context:", context.strip())
    print("\nQuestions and Answers:")
    
    # Get answers
    results = qa.answer_questions(questions, context)
    
    # Display results
    for result in results:
        print(f"\nQ: {result['question']}")
        print(f"A: {result['answer']}")
        print(f"Confidence: {result['confidence']:.3f}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('qa_results.csv', index=False)
    print("\nResults saved to qa_results.csv")

if __name__ == "__main__":
    main() 