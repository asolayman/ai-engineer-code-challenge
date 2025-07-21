AI Engineer Code Challenge Documentation
======================================

Welcome to the AI Engineer Code Challenge documentation. This project implements a **document-based question-answering system** that can ingest PDF documents and answer questions using local language models.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api
   configuration
   cli
   architecture
   testing
   contributing

Features
--------

* **Local Operation**: Works completely offline without internet connection
* **Multiple PDF Engines**: Support for PyMuPDF, pdfminer, and pdfplumber
* **Configurable Chunking**: Adjustable chunk size and overlap parameters
* **Vector Similarity Search**: FAISS-based retrieval for efficient searching
* **Local LLM Integration**: Support for transformers and llama-cpp backends
* **Modular Architecture**: Clean separation of concerns with extensible components
* **Comprehensive Testing**: Unit and integration tests with high coverage
* **Production Ready**: Robust error handling and logging

Quick Start
-----------

.. code-block:: bash

   # Install dependencies
   pip install -r requirements.txt
   
   # Ingest documents
   python main.py --mode ingest --documents ./data/
   
   # Ask a question
   python main.py --mode query --query "What are the key features?"

Architecture
-----------

The system follows a modular RAG (Retrieval-Augmented Generation) architecture:

1. **Document Ingestion**: PDF processing and text chunking
2. **Embedding Generation**: Vector embeddings using sentence transformers
3. **Index Storage**: FAISS vector database for efficient retrieval
4. **Query Processing**: Semantic search and context retrieval
5. **Answer Generation**: Local LLM for answer synthesis

Technology Stack
---------------

* **PDF Processing**: PyMuPDF, pdfminer.six, pdfplumber
* **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
* **Vector Database**: FAISS (IndexFlatIP)
* **Language Models**: transformers, llama-cpp-python
* **Testing**: pytest, pytest-cov
* **Code Quality**: ruff, black, isort
* **Documentation**: Sphinx, Read the Docs theme

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 