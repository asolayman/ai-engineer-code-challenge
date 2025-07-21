Installation Guide
=================

This guide will help you install and set up the Document-Based Question Answering System.

Prerequisites
------------

* **Python 3.10+**: The system requires Python 3.10 or higher
* **8GB+ RAM**: For local LLM models and embedding generation
* **5GB+ disk space**: For models and indexes
* **Git**: For cloning the repository

System Requirements
------------------

* **Operating System**: Windows, macOS, or Linux
* **Memory**: Minimum 8GB RAM (16GB+ recommended for large models)
* **Storage**: 5GB+ free disk space
* **Network**: Internet connection for initial model downloads

Installation Steps
-----------------

1. **Clone the Repository**

   .. code-block:: bash

      git clone <repository-url>
      cd ai-engineer-code-challenge

2. **Create Virtual Environment** (Recommended)

   .. code-block:: bash

      # Create virtual environment
      python -m venv venv

      # Activate virtual environment
      # On Windows:
      venv\Scripts\activate
      # On macOS/Linux:
      source venv/bin/activate

3. **Install Dependencies**

   .. code-block:: bash

      pip install -r requirements.txt

4. **Download Models** (Optional)

   .. code-block:: bash

      # Create models directory
      mkdir models

      # Download a GGUF model for llama-cpp (optional)
      # wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf -O models/mistral-7b-instruct.gguf

5. **Verify Installation**

   .. code-block:: bash

      # Test the installation
      python main.py --help

Installation Options
-------------------

Standard Installation
~~~~~~~~~~~~~~~~~~~~

The standard installation includes all core dependencies:

.. code-block:: bash

   pip install -r requirements.txt

Minimal Installation
~~~~~~~~~~~~~~~~~~~

For minimal installation (without optional dependencies):

.. code-block:: bash

   pip install python-dotenv PyYAML argparse PyMuPDF sentence-transformers faiss-cpu numpy transformers torch accelerate pytest pytest-cov pytest-mock ruff black structlog tqdm psutil

GPU Support
~~~~~~~~~~

For GPU acceleration (optional):

.. code-block:: bash

   # Install PyTorch with CUDA support
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

   # Install FAISS with GPU support
   pip install faiss-gpu

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~

For development with documentation:

.. code-block:: bash

   pip install -r requirements.txt
   pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints myst-parser

Configuration
------------

1. **Create Configuration File**

   The system uses `config.yaml` for configuration. A sample configuration is provided:

   .. code-block:: yaml

      # PDF Processing
      pdf:
        engine: "pymupdf"
        chunk_size: 1000
        chunk_overlap: 200

      # Embedding Model
      embedding:
        model_name: "all-MiniLM-L6-v2"
        similarity_threshold: 0.7
        top_k: 5

      # LLM Configuration
      llm:
        backend: "transformers"
        model_path: "microsoft/DialoGPT-medium"
        temperature: 0.2
        max_tokens: 1024

2. **Set Environment Variables** (Optional)

   Create a `.env` file for sensitive configuration:

   .. code-block:: bash

      # OpenAI API (if using OpenAI backend)
      OPENAI_API_KEY=your_api_key_here

      # Custom model paths
      LLM_MODEL_PATH=./models/custom-model.gguf
      EMBEDDING_MODEL_PATH=./models/custom-embedding

Troubleshooting
--------------

Common Installation Issues
~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Memory Issues**

   If you encounter memory issues during installation:

   .. code-block:: bash

      # Use pip with memory optimization
      pip install --no-cache-dir -r requirements.txt

2. **Compilation Issues**

   For compilation issues with llama-cpp-python:

   .. code-block:: bash

      # Install with specific compiler flags
      CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python

3. **CUDA Issues**

   If you have CUDA issues:

   .. code-block:: bash

      # Install CPU-only version
      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

4. **Permission Issues**

   For permission issues on Linux/macOS:

   .. code-block:: bash

      # Use user installation
      pip install --user -r requirements.txt

Verification
-----------

After installation, verify the setup:

.. code-block:: bash

   # Test basic functionality
   python main.py --help

   # Test configuration loading
   python -c "import yaml; yaml.safe_load(open('config.yaml'))"

   # Test imports
   python -c "from src.ingest import DocumentIngester; print('✓ Imports working')"

Next Steps
----------

After successful installation:

1. **Read the Quick Start Guide**: :doc:`quickstart`
2. **Configure the System**: :doc:`configuration`
3. **Try the Examples**: :doc:`user_guide/examples`

For more detailed information, see the :doc:`user_guide/index`. 