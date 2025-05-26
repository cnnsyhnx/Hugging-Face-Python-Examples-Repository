"""
Text Generation with Different Models

This module demonstrates text generation using various Hugging Face models.
"""

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List, Dict
import pandas as pd

class TextGenerator:
    """Text generation system using Hugging Face."""
    
    def __init__(self, model_name: str = "gpt2"):
        """Initialize the text generation pipeline."""
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_text(
        self,
        prompt: str,
        max_length: int = 100,
        num_return_sequences: int = 1,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50
    ) -> List[Dict]:
        """Generate text from a prompt."""
        # Encode the prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Generate text
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=True
        )
        
        # Decode and return results
        results = []
        for output in outputs:
            generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
            results.append({
                'prompt': prompt,
                'generated_text': generated_text,
                'length': len(generated_text.split())
            })
        
        return results

def create_sample_prompts():
    """Create sample prompts for demonstration."""
    return [
        "The future of artificial intelligence",
        "Once upon a time in a galaxy",
        "The most important thing to remember about machine learning is",
        "In the world of natural language processing"
    ]

def main():
    """Run text generation example."""
    print("✍️ Text Generation Example")
    print("-" * 40)
    
    # Create text generator
    generator = TextGenerator()
    
    # Get sample prompts
    prompts = create_sample_prompts()
    
    # Generate text for each prompt
    all_results = []
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        results = generator.generate_text(
            prompt,
            max_length=150,
            num_return_sequences=2,
            temperature=0.8
        )
        
        for i, result in enumerate(results, 1):
            print(f"\nGenerated text {i}:")
            print(result['generated_text'])
            print(f"Length: {result['length']} words")
        
        all_results.extend(results)
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('generation_results.csv', index=False)
    print("\nResults saved to generation_results.csv")

if __name__ == "__main__":
    main() 