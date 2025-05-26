# 🤗 Hugging Face Python Examples Repository

A comprehensive collection of practical examples and tutorials for using Hugging Face libraries in Python.

## 📚 Contents

- Basic pipeline examples
- Text classification
- Question answering
- Text generation
- Sentiment analysis
- Custom model training
- Interactive notebooks
- Utility functions

## 🚀 Getting Started

1. Clone the repository:
```bash
git clone https://github.com/yourusername/huggingface-examples.git
cd huggingface-examples
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run examples:
```bash
python examples/basic_pipelines.py
python examples/text_classification.py
python examples/sentiment_analysis.py
```

4. For development:
```bash
pip install -e .
```

5. Run tests:
```bash
python -m pytest tests/
```

## 📁 Project Structure

```
/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── examples/
│   ├── __init__.py
│   ├── basic_pipelines.py
│   ├── text_classification.py
│   ├── question_answering.py
│   ├── text_generation.py
│   ├── sentiment_analysis.py
│   └── custom_training.py
├── notebooks/
│   ├── 01_getting_started.ipynb
│   ├── 02_fine_tuning.ipynb
│   └── 03_deployment.ipynb
├── utils/
│   ├── __init__.py
│   ├── data_helpers.py
│   └── model_helpers.py
└── tests/
    ├── __init__.py
    └── test_examples.py
```

## 🛠️ Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.35+
- Other dependencies listed in requirements.txt

## 📝 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 