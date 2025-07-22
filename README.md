# AI Engineer Code Challenge

[![CI/CD](https://github.com/asolayman/ai-engineer-code-challenge/actions/workflows/ci.yml/badge.svg)](https://github.com/solayman/ai-engineer-code-challenge/actions/workflows/ci.yml)
[![Documentation](https://github.com/asolayman/ai-engineer-code-challenge/actions/workflows/docs.yml/badge.svg)](https://asolayman.github.io/ai-engineer-code-challenge/)
[![Code Coverage](https://codecov.io/gh/asolayman/ai-engineer-code-challenge/branch/main/graph/badge.svg)](https://codecov.io/gh/asolayman/ai-engineer-code-challenge)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A **document-based question-answering system** that ingests PDF documents and enables natural language queries using local language models. The system works completely offline and provides accurate, context-aware answers based on document content.

## Features

- **Fully Offline**: Works without internet using local models
- **Multi-Engine PDF Processing**: Support for PyMuPDF, pdfminer, and pdfplumber
- **Configurable Chunking**: Adjustable chunk size and overlap parameters
- **Vector Similarity Search**: FAISS-based retrieval for efficient searching
- **Local LLM Integration**: Support for transformers and llama-cpp backends
- **Modular Architecture**: Clean separation of concerns with extensible components
- **Comprehensive Testing**: Unit and integration tests with high coverage
- **Production Ready**: Robust error handling and logging

## Requirements

- Python 3.10+
- 8GB+ RAM (for LLM inference)
- 4GB+ disk space (for models and indexes)

## Installation

### Quick Start

```bash
# Clone the repository
git clone https://github.com/asolayman/ai-engineer-code-challenge.git
cd ai-engineer-code-challenge

# Install dependencies
pip install -r requirements.txt

# Download a local LLM model (optional)
# Place your GGUF model in the models/ directory
mkdir -p models
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf -o ./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf

```


## Quick Start

### 1. Prepare Your Documents

Place your PDF documents in the `data/` directory:

```bash
mkdir -p data
cp your-documents/*.pdf data/
```

### 2. Ingest Documents

```bash
python main.py --mode ingest --documents ./data/
```

This will:
- Extract text from PDF documents
- Chunk the text into smaller pieces
- Generate embeddings for each chunk
- Store embeddings in a FAISS index

### 3. Ask Questions

```bash
python main.py --mode query --query "What are the key features of this system?"
```

### 4. Advanced Usage

```bash
# With verbose output
python main.py --mode query --query "Your question" --verbose

# Override configuration
python main.py --mode query --query "Your question" --similarity-threshold 0.5 --top-k 10

# Custom chunking parameters
python main.py --mode ingest --documents ./data/ --chunk-size 1500 --chunk-overlap 300
```

## Configuration

The system uses `config.yaml` for configuration. Key settings:

```yaml
# PDF Processing
pdf:
  engine: "pymupdf"  # Options: "pymupdf", "pdfminer", "pdfplumber"
  chunk_size: 1000
  chunk_overlap: 200

# Embeddings
embedding:
  model: "all-MiniLM-L6-v2"
  top_k: 5
  similarity_threshold: 0.7

# LLM Configuration
llm:
  backend: "llama-cpp"  # Options: "transformers", "llama-cpp", "openai"
  model_path: "./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
  temperature: 0.1
  max_tokens: 200
```

## Architecture

The system follows a modular RAG (Retrieval-Augmented Generation) architecture:

1. **Document Ingestion**: PDF processing and text chunking
2. **Embedding Generation**: Vector embeddings using sentence transformers
3. **Index Storage**: FAISS vector database for efficient retrieval
4. **Query Processing**: Semantic search and context retrieval
5. **Answer Generation**: Local LLM for answer synthesis

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test categories
pytest -m unit tests/
pytest -m integration tests/

# Run linting
ruff check src/ tests/
black --check src/ tests/
```

## Documentation

- **[Full Documentation](https://asolayman.github.io/ai-engineer-code-challenge/)**
- **[API Reference](https://asolayman.github.io/ai-engineer-code-challenge/api.html)**


### Development Commands

```bash
# Format code
make format

# Run linting
make lint

# Run tests
make test

# Build documentation
make docs

# Run all checks
make ci
```


##  Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce chunk size or use smaller models
2. **Slow Performance**: Use quantized models or GPU acceleration
3. **No Results**: Lower similarity threshold or increase top_k
4. **Model Loading**: Ensure model files are in the correct location

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python main.py --mode query --query "Your question" --verbose
```