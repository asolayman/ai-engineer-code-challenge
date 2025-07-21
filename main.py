#!/usr/bin/env python3
"""
Document-based Question Answering System

A local, modular RAG (retrieval-augmented generation) system that processes
PDF documents and enables natural language queries.

Usage:
    python main.py --mode ingest --documents ./data/
    python main.py --mode query --query "What is Consult+ prediction for Tesla stock?"
"""

import argparse
import logging
import sys
import copy
from pathlib import Path
from typing import Optional, Dict, Any

import yaml
from dotenv import load_dotenv

# Import utility functions
from src.utils import setup_logging, get_logger, log_system_info

# Load environment variables
load_dotenv()

# Get logger
logger = get_logger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file {config_path} not found")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        sys.exit(1)


def validate_ingest_args(args: argparse.Namespace) -> None:
    """
    Validate arguments for ingest mode.
    
    Args:
        args: Parsed command line arguments
        
    Raises:
        ValueError: If validation fails
    """
    documents_path = Path(args.documents)
    if not documents_path.exists():
        raise ValueError(f"Documents directory {documents_path} does not exist")
    
    if not documents_path.is_dir():
        raise ValueError(f"{documents_path} is not a directory")
    
    # Check if directory contains PDF files
    pdf_files = list(documents_path.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {documents_path}")
    
    logger.info(f"Found {len(pdf_files)} PDF files in {documents_path}")


def validate_query_args(args: argparse.Namespace) -> None:
    """
    Validate arguments for query mode.
    
    Args:
        args: Parsed command line arguments
        
    Raises:
        ValueError: If validation fails
    """
    if not args.query or not args.query.strip():
        raise ValueError("Query cannot be empty")
    
    # Check if index exists
    index_path = Path("index")
    if not index_path.exists():
        raise ValueError("Index directory does not exist. Please run ingest mode first.")
    
    logger.info(f"Query: {args.query}")


def merge_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Merge CLI arguments with configuration, CLI args take precedence.
    
    Args:
        config: Configuration dictionary
        args: Parsed command line arguments
        
    Returns:
        Updated configuration dictionary
    """
    # Create a deep copy to avoid modifying the original
    merged_config = copy.deepcopy(config)
    
    # Merge PDF processing settings
    if args.chunk_size is not None:
        if "pdf" not in merged_config:
            merged_config["pdf"] = {}
        merged_config["pdf"]["chunk_size"] = args.chunk_size
    
    if args.chunk_overlap is not None:
        if "pdf" not in merged_config:
            merged_config["pdf"] = {}
        merged_config["pdf"]["chunk_overlap"] = args.chunk_overlap
    
    if args.pdf_engine is not None:
        if "pdf" not in merged_config:
            merged_config["pdf"] = {}
        merged_config["pdf"]["engine"] = args.pdf_engine
    
    # Merge embedding settings
    if args.embedding_model is not None:
        if "embedding" not in merged_config:
            merged_config["embedding"] = {}
        merged_config["embedding"]["model_name"] = args.embedding_model
    
    if args.top_k is not None:
        if "embedding" not in merged_config:
            merged_config["embedding"] = {}
        merged_config["embedding"]["top_k"] = args.top_k
    
    if args.similarity_threshold is not None:
        if "embedding" not in merged_config:
            merged_config["embedding"] = {}
        merged_config["embedding"]["similarity_threshold"] = args.similarity_threshold
    
    # Merge LLM settings
    if args.llm_backend is not None:
        if "llm" not in merged_config:
            merged_config["llm"] = {}
        merged_config["llm"]["backend"] = args.llm_backend
    
    if args.llm_model is not None:
        if "llm" not in merged_config:
            merged_config["llm"] = {}
        merged_config["llm"]["model_path"] = args.llm_model
    
    if args.temperature is not None:
        if "llm" not in merged_config:
            merged_config["llm"] = {}
        merged_config["llm"]["temperature"] = args.temperature
    
    if args.max_tokens is not None:
        if "llm" not in merged_config:
            merged_config["llm"] = {}
        merged_config["llm"]["max_tokens"] = args.max_tokens
    
    return merged_config


def main() -> None:
    """
    Main entry point for the document-based question answering system.
    
    Handles command line argument parsing and routes to appropriate
    functionality based on the selected mode.
    """
    parser = argparse.ArgumentParser(
        description="Document-based Question Answering System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode ingest --documents ./data/
  python main.py --mode query --query "What is Consult+ prediction for Tesla stock?"
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["ingest", "query"],
        required=True,
        help="Operation mode: ingest documents or query the system"
    )
    
    parser.add_argument(
        "--documents",
        type=str,
        help="Path to directory containing PDF documents (required for ingest mode)"
    )
    
    parser.add_argument(
        "--query",
        type=str,
        help="Natural language query (required for query mode)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        help="Size of text chunks for processing (overrides config)"
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        help="Overlap between text chunks (overrides config)"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        help="Number of top similar chunks to retrieve (overrides config)"
    )

    parser.add_argument(
        "--similarity-threshold",
        type=float,
        help="Minimum similarity threshold for chunk selection (overrides config)"
    )

    parser.add_argument(
        "--embedding-model",
        type=str,
        help="Embedding model to use (overrides config)"
    )

    parser.add_argument(
        "--llm-backend",
        type=str,
        choices=["transformers", "llama-cpp", "openai"],
        help="LLM backend to use (overrides config)"
    )

    parser.add_argument(
        "--llm-model",
        type=str,
        help="LLM model path or name (overrides config)"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        help="LLM temperature (overrides config)"
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Maximum tokens for LLM generation (overrides config)"
    )

    parser.add_argument(
        "--pdf-engine",
        type=str,
        choices=["pymupdf", "pdfplumber", "pdfminer"],
        help="PDF processing engine (overrides config)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set up logging based on configuration
    try:
        # Load configuration for logging setup
        config = load_config(args.config)
        system_config = config.get("system", {})
        
        # Set up logging
        setup_logging(
            log_level=system_config.get("log_level", "INFO"),
            log_file=system_config.get("log_file", None)
        )
        
        # Log system information
        log_system_info(logger)
        
    except Exception as e:
        # Fallback to basic logging if config loading fails
        setup_logging(log_level="INFO")
        logger.warning(f"Could not load configuration for logging setup: {e}")
    
    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Merge CLI arguments with configuration
        config = merge_config_with_args(config, args)
        
        # Validate arguments based on mode
        if args.mode == "ingest":
            validate_ingest_args(args)
            logger.info("Starting document ingestion...")
            from src.ingest import ingest_documents
            ingest_documents(args.documents, config, args)
            
            # After ingestion, create embeddings
            logger.info("Creating embeddings from ingested chunks...")
            from src.embed import create_embeddings_from_chunks_file
            chunks_file = Path(config.get("storage", {}).get("index_dir", "./index")) / "chunks.json"
            output_path = Path(config.get("storage", {}).get("index_dir", "./index"))
            create_embeddings_from_chunks_file(chunks_file, config, output_path)
            
        elif args.mode == "query":
            validate_query_args(args)
            logger.info("Starting query processing...")
            from src.query import process_query, format_query_output
            from src.llm import generate_answer_from_query, format_llm_response
            
            # Process query to get relevant chunks
            result = process_query(args.query, config, args)
            
            # Generate answer using LLM
            logger.info("Generating answer using LLM...")
            answer = generate_answer_from_query(args.query, result, config)
            
            # Format output
            if args.verbose:
                # Show both chunks and LLM answer
                chunks_output = format_query_output(result, verbose=True)
                llm_output = format_llm_response(
                    type('obj', (object,), {
                        'answer': answer,
                        'prompt_tokens': 0,
                        'response_tokens': 0,
                        'generation_time_ms': 0.0,
                        'model_used': 'llm'
                    }), verbose=True
                )
                print(f"{chunks_output}\n\n{llm_output}")
            else:
                # Show only the answer
                print(answer)
            
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 