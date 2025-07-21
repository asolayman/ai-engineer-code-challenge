API Reference
============

This section provides detailed documentation for the AI Engineer Code Challenge API.

Core Modules
------------

.. automodule:: src.ingest
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: src.embed
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: src.query
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: src.llm
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: src.utils
   :members:
   :undoc-members:
   :show-inheritance:

Data Structures
---------------

.. automodule:: src.ingest
   :members: DocumentChunk, ChunkMetadata
   :undoc-members:
   :show-inheritance:

.. automodule:: src.embed
   :members: EmbeddingConfig, FAISSIndex
   :undoc-members:
   :show-inheritance:

.. automodule:: src.query
   :members: QueryResult
   :undoc-members:
   :show-inheritance:

.. automodule:: src.llm
   :members: LLMConfig, LLMResponse
   :undoc-members:
   :show-inheritance:

Configuration
-------------

The system uses YAML configuration files for all settings. Here's the structure:

.. code-block:: yaml

   # PDF Processing Configuration
   pdf:
     engine: "pymupdf"  # Options: "pymupdf", "pdfminer", "pdfplumber"
     chunk_size: 1000
     chunk_overlap: 200
   
   # Embedding Configuration
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
     top_p: 0.9
     repeat_penalty: 1.1
     context_window: 4096
   
   # Storage Configuration
   storage:
     index_dir: "./index"
     chunk_dir: "./index/chunks"
   
   # System Configuration
   system:
     log_level: "INFO"
     batch_size: 100
     max_workers: 4

Command Line Interface
---------------------

The main entry point provides a command-line interface:

.. code-block:: bash

   # Ingest documents
   python main.py --mode ingest --documents ./data/
   
   # Query the system
   python main.py --mode query --query "What are the key features?"
   
   # With verbose output
   python main.py --mode query --query "Your question" --verbose
   
   # Override configuration
   python main.py --mode query --query "Your question" --similarity-threshold 0.5

Available CLI Options:

.. code-block:: text

   --mode: Choose between 'ingest' or 'query'
   --documents: Path to documents directory (for ingest mode)
   --query: Your question (for query mode)
   --verbose: Enable verbose output
   --similarity-threshold: Override similarity threshold
   --top-k: Override number of chunks to retrieve
   --chunk-size: Override chunk size for ingestion
   --chunk-overlap: Override chunk overlap for ingestion
   --embedding-model: Override embedding model
   --llm-backend: Override LLM backend
   --llm-model: Override LLM model path
   --temperature: Override LLM temperature
   --max-tokens: Override LLM max tokens

Examples
--------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from src.ingest import ingest_documents
   from src.query import process_query
   from src.llm import generate_answer_from_query
   
   # Ingest documents
   config = load_config("config.yaml")
   ingest_documents("./data/", config, args)
   
   # Query the system
   result = process_query("What is this about?", config, args)
   answer = generate_answer_from_query("What is this about?", result, config)

Advanced Usage
~~~~~~~~~~~~~

.. code-block:: python

   from src.embed import EmbeddingPipeline
   from src.query import QueryProcessor
   from src.llm import LLMInterface
   
   # Custom embedding pipeline
   embedding_pipeline = EmbeddingPipeline(config)
   embedding_pipeline.create_embeddings_from_chunks(chunks)
   
   # Custom query processing
   query_processor = QueryProcessor(config, index_path)
   result = query_processor.process_query("Your question", top_k=10, similarity_threshold=0.8)
   
   # Custom LLM interface
   llm_interface = LLMInterface(config)
   response = llm_interface.generate_answer("Your question", result)

Error Handling
-------------

The system provides comprehensive error handling:

.. code-block:: python

   try:
       result = process_query("Your question", config, args)
   except ValueError as e:
       print(f"Configuration error: {e}")
   except FileNotFoundError as e:
       print(f"File not found: {e}")
   except Exception as e:
       print(f"Unexpected error: {e}")

Performance Optimization
-----------------------

For optimal performance:

1. **Use appropriate chunk sizes**: 1000-2000 characters work well for most documents
2. **Adjust similarity threshold**: 0.7-0.8 provides good balance
3. **Batch processing**: Use batch_size in system config for large datasets
4. **Model selection**: Choose quantized models for faster inference
5. **Hardware utilization**: Use GPU if available for LLM inference

Monitoring and Logging
---------------------

The system provides comprehensive logging:

.. code-block:: python

   import logging
   
   # Configure logging
   logging.basicConfig(level=logging.INFO)
   
   # Monitor performance
   from src.utils import log_performance, log_memory_usage
   
   @log_performance
   def your_function():
       # Your code here
       pass

Testing
-------

The system includes comprehensive tests:

.. code-block:: bash

   # Run all tests
   pytest tests/
   
   # Run with coverage
   pytest --cov=src tests/
   
   # Run specific test categories
   pytest -m unit tests/
   pytest -m integration tests/ 