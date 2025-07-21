API Reference
============

This section contains the complete API reference for the Document-Based Question Answering System.

Core Modules
-----------

.. toctree::
   :maxdepth: 2

   main
   ingest
   embed
   query
   llm
   utils

CLI Interface
------------

The main entry point for the system is the `main.py` module, which provides a command-line interface for document ingestion and querying.

.. automodule:: main
   :members:
   :undoc-members:
   :show-inheritance:

Document Ingestion
-----------------

The `ingest` module handles PDF document processing, text extraction, and chunking.

.. automodule:: src.ingest
   :members:
   :undoc-members:
   :show-inheritance:

Embedding Generation
-------------------

The `embed` module manages embedding model loading, vector generation, and FAISS indexing.

.. automodule:: src.embed
   :members:
   :undoc-members:
   :show-inheritance:

Query Processing
---------------

The `query` module handles query embedding, similarity search, and result ranking.

.. automodule:: src.query
   :members:
   :undoc-members:
   :show-inheritance:

LLM Interface
-------------

The `llm` module provides interfaces for different LLM backends and answer generation.

.. automodule:: src.llm
   :members:
   :undoc-members:
   :show-inheritance:

Utilities
---------

The `utils` module contains utility functions for logging, performance monitoring, and system information.

.. automodule:: src.utils
   :members:
   :undoc-members:
   :show-inheritance:

Data Structures
--------------

Core data structures used throughout the system:

.. automodule:: src.ingest
   :members: DocumentChunk, ChunkMetadata
   :undoc-members:
   :show-inheritance:

.. automodule:: src.query
   :members: QueryResult
   :undoc-members:
   :show-inheritance:

Configuration
------------

The system uses YAML configuration files. See :doc:`../configuration` for detailed configuration options.

Error Handling
-------------

The system provides comprehensive error handling for various scenarios:

* **FileNotFoundError**: When documents or models are not found
* **ValueError**: When configuration is invalid
* **RuntimeError**: When models fail to load or process
* **MemoryError**: When system runs out of memory

Performance Considerations
------------------------

* **Memory Usage**: Monitor memory usage with `log_memory_usage()`
* **Batch Processing**: Use batch processing for large datasets
* **Caching**: Enable caching for frequently accessed data
* **Optimization**: Use `optimize_memory()` for memory cleanup

Examples
--------

See :doc:`../user_guide/examples` for practical usage examples. 