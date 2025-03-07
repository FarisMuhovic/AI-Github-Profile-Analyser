# GitHub Repository Analyzer

## Overview

The **GitHub Repository Analyzer** is a Python-based tool designed to analyze public repositories of a specified GitHub user. It fetches code files from repositories, processes them using natural language processing (NLP) techniques, and generates insights about programming language usage and topic distribution. The tool leverages **Ollama** for enhanced NLP capabilities.

## Features

- **File Retrieval**: Extracts files from public repositories using the GitHub API.
- **Text Preprocessing**: Cleans text by removing stopwords, punctuation, and numbers.
- **Keyword Extraction**: Uses Count Vectorizer and TF-IDF Vectorizer to generate vocabulary arrays.
- **Topic Modeling**: Identifies topics using Latent Dirichlet Allocation (LDA).
- **Programming Language Detection**: Infers dominant programming languages based on extracted keywords.

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/github-repo-analyzer.git
cd github-repo-analyzer

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python main.py
---
Afterwards, you will be prompted in the terminal with all the necessary information you have to enter.
```

Replace `<github-username>` with the GitHub profile you want to analyze.

## Methodology

1. **File Retrieval**: Extracts files using the GitHub API.
2. **Text Preprocessing**: Cleans the text using tokenization and stopword removal.
3. **TF-IDF Vectorization**: Calculates term frequency-inverse document frequency scores.
4. **Topic Modeling**: Applies LDA to detect topics in the dataset.

## Limitations

- Only works with public repositories.
- Results may be inaccurate for repositories with minimal textual content.
- Topic modeling effectiveness depends on the quality of extracted text.

## Future Enhancements

- Extend support for private repositories via authentication.
- Improve topic coherence by refining the LDA model.
- Add visualization tools for enhanced insights.

## Research Paper

For detailed research and methodology, please refer to our research paper available in the root directory: **[Research_Paper.pdf](Research_Paper.pdf)**.
