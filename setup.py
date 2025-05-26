from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="huggingface-examples",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Practical Hugging Face examples and tutorials",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/huggingface-examples",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "tokenizers>=0.15.0",
        "torch>=2.0.0",
        "evaluate>=0.4.0",
        "accelerate>=0.24.0",
    ],
) 