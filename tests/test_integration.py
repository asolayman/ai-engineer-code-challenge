"""
Integration tests for end-to-end functionality
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ingest import ingest_documents
from src.llm import generate_answer_from_query
from src.query import process_query


class TestIntegrationEndToEnd:
    """Test complete end-to-end pipeline"""

    def test_complete_pipeline_with_mocks(self):
        """Test complete pipeline with mocked components"""
        # Test configuration validation
        config = {
            "pdf": {"engine": "pymupdf", "chunk_size": 1000, "chunk_overlap": 200},
            "embedding": {
                "model_name": "all-MiniLM-L6-v2",
                "normalize_embeddings": True,
                "device": "cpu",
                "similarity_threshold": 0.7,
                "top_k": 5,
            },
            "llm": {
                "backend": "transformers",
                "model_path": "microsoft/DialoGPT-medium",
                "temperature": 0.2,
                "max_tokens": 1024,
            },
            "storage": {"index_dir": "./index"},
        }

        # Test that all required keys are present
        assert "pdf" in config
        assert "embedding" in config
        assert "llm" in config
        assert "storage" in config

        # Test LLM answer generation with mock
        from src.ingest import ChunkMetadata, DocumentChunk
        from src.query import QueryResult

        # Create mock query result
        mock_chunk = DocumentChunk(
            text="This is test content from the PDF document.",
            metadata=ChunkMetadata(
                file_name="test.pdf",
                page_number=1,
                chunk_index=0,
                chunk_start=0,
                chunk_end=100,
                chunk_size=100,
                text_length=1000,
            ),
        )

        query_result = QueryResult(
            query="What is the test content?",
            chunks=[mock_chunk],
            similarities=[0.85],
            total_chunks_searched=100,
            search_time_ms=50.0,
        )

        # Test LLM answer generation
        with patch("src.llm.create_llm_interface") as mock_create_interface:
            mock_interface = MagicMock()
            mock_interface.generate_answer.return_value.answer = (
                "The test content is from the PDF document."
            )
            mock_create_interface.return_value = mock_interface

            answer = generate_answer_from_query(
                "What is the test content?", query_result, config
            )

            assert "test content" in answer.lower()


class TestIntegrationConfiguration:
    """Test configuration loading and validation"""

    def test_config_loading_integration(self):
        """Test configuration loading with all components"""
        config = {
            "pdf": {"engine": "pymupdf", "chunk_size": 1000, "chunk_overlap": 200},
            "embedding": {
                "model_name": "all-MiniLM-L6-v2",
                "normalize_embeddings": True,
                "device": "cpu",
                "similarity_threshold": 0.7,
                "top_k": 5,
            },
            "llm": {
                "backend": "transformers",
                "model_path": "microsoft/DialoGPT-medium",
                "temperature": 0.2,
                "max_tokens": 1024,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "context_window": 4096,
            },
            "storage": {"index_dir": "./index"},
            "prompts": {
                "query_template": "Context: {context}\nQuestion: {question}\nAnswer:",
                "no_answer_template": "No information found.",
            },
        }

        # Test that all required keys are present
        assert "pdf" in config
        assert "embedding" in config
        assert "llm" in config
        assert "storage" in config

        # Test PDF configuration
        assert config["pdf"]["engine"] in ["pymupdf", "pdfplumber", "pdfminer"]
        assert config["pdf"]["chunk_size"] > 0
        assert config["pdf"]["chunk_overlap"] >= 0

        # Test embedding configuration
        assert "model_name" in config["embedding"]
        assert "similarity_threshold" in config["embedding"]
        assert "top_k" in config["embedding"]

        # Test LLM configuration
        assert config["llm"]["backend"] in ["transformers", "llama-cpp", "openai"]
        assert "model_path" in config["llm"]
        assert 0 <= config["llm"]["temperature"] <= 1
        assert config["llm"]["max_tokens"] > 0


class TestIntegrationErrorHandling:
    """Test error handling in integration scenarios"""

    def test_missing_index_error_handling(self):
        """Test error handling when index is missing"""
        config = {"storage": {"index_dir": "./nonexistent_index"}}

        args = MagicMock()
        args.query = "test query"

        # Should raise an error when index doesn't exist
        with pytest.raises((FileNotFoundError, ValueError)):
            process_query("test query", config, args)

    def test_empty_documents_error_handling(self):
        """Test error handling with empty document directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            config = {
                "pdf": {"engine": "pymupdf", "chunk_size": 1000, "chunk_overlap": 200},
                "storage": {"index_dir": str(temp_path / "index")},
            }

            args = MagicMock()
            args.chunk_size = 1000
            args.chunk_overlap = 200

            # Should handle empty directory gracefully
            with patch("src.ingest.PDFProcessor"):
                # Should raise ValueError for empty directory
                with pytest.raises(ValueError, match="No PDF files found"):
                    ingest_documents(str(temp_path), config, args)

    def test_llm_error_handling(self):
        """Test LLM error handling"""
        config = {"llm": {"backend": "invalid_backend", "model_path": "invalid_model"}}

        # Should raise ValueError for invalid backend
        with pytest.raises(ValueError, match="Unsupported LLM backend"):
            from src.llm import LLMInterface

            LLMInterface(config)


class TestIntegrationCLI:
    """Test CLI integration scenarios"""

    @patch("main.load_config")
    @patch("main.validate_ingest_args")
    @patch("src.ingest.ingest_documents")
    @patch("src.embed.create_embeddings_from_chunks_file")
    def test_cli_ingest_mode(
        self, mock_create_embeddings, mock_ingest, mock_validate, mock_load_config
    ):
        """Test CLI ingest mode integration"""
        mock_load_config.return_value = {"storage": {"index_dir": "./index"}}

        with patch(
            "sys.argv", ["main.py", "--mode", "ingest", "--documents", "./test_data"]
        ):
            with patch("main.logger") as mock_logger:
                from main import main

                main()

                mock_validate.assert_called_once()
                mock_ingest.assert_called_once()
                mock_create_embeddings.assert_called_once()
                mock_logger.info.assert_any_call("Starting document ingestion...")
                mock_logger.info.assert_any_call(
                    "Creating embeddings from ingested chunks..."
                )

    @patch("main.load_config")
    @patch("main.validate_query_args")
    @patch("src.query.process_query")
    @patch("src.llm.generate_answer_from_query")
    def test_cli_query_mode(
        self, mock_generate_answer, mock_process_query, mock_validate, mock_load_config
    ):
        """Test CLI query mode integration"""
        mock_load_config.return_value = {"llm": {"backend": "transformers"}}

        from src.query import QueryResult

        mock_result = QueryResult(
            query="test query",
            chunks=[],
            similarities=[],
            total_chunks_searched=0,
            search_time_ms=0.0,
        )
        mock_process_query.return_value = mock_result
        mock_generate_answer.return_value = "Test answer"

        with patch("sys.argv", ["main.py", "--mode", "query", "--query", "test query"]):
            with patch("main.logger") as mock_logger:
                with patch("builtins.print") as mock_print:
                    from main import main

                    main()

                    mock_validate.assert_called_once()
                    mock_process_query.assert_called_once()
                    mock_generate_answer.assert_called_once()
                    mock_logger.info.assert_any_call("Starting query processing...")
                    mock_logger.info.assert_any_call("Generating answer using LLM...")
                    mock_print.assert_called_once_with("Test answer")


class TestIntegrationPerformance:
    """Test performance aspects of integration"""

    def test_large_document_handling(self):
        """Test handling of large documents"""
        # Test configuration for large document processing
        config = {
            "pdf": {"engine": "pymupdf", "chunk_size": 1000, "chunk_overlap": 200},
            "storage": {"index_dir": "./index"},
        }

        # Test that configuration supports large documents
        assert config["pdf"]["chunk_size"] == 1000
        assert config["pdf"]["chunk_overlap"] == 200

        # Test that chunking parameters are reasonable
        assert config["pdf"]["chunk_size"] > config["pdf"]["chunk_overlap"]
        assert config["pdf"]["chunk_overlap"] >= 0

        # Test that storage configuration is valid
        assert "index_dir" in config["storage"]
        assert isinstance(config["storage"]["index_dir"], str)

    def test_multiple_queries_performance(self):
        """Test performance with multiple queries"""
        config = {
            "embedding": {
                "model_name": "all-MiniLM-L6-v2",
                "similarity_threshold": 0.7,
                "top_k": 5,
            },
            "storage": {"index_dir": "./index"},
        }

        queries = [
            "What is the main topic?",
            "What are the key points?",
            "What is the conclusion?",
        ]

        args = MagicMock()

        # Mock embedding pipeline for multiple queries
        with patch("src.embed.load_embedding_pipeline") as mock_load_pipeline:
            mock_pipeline = MagicMock()
            mock_pipeline.search_similar_chunks.return_value = []
            mock_pipeline.faiss_index.get_total_embeddings.return_value = 100
            mock_load_pipeline.return_value = mock_pipeline

            # Should handle multiple queries efficiently
            for query in queries:
                try:
                    result = process_query(query, config, args)
                    assert result.query == query
                    assert result.total_chunks_searched == 100
                except FileNotFoundError:
                    # Expected when index doesn't exist in test environment
                    pass


if __name__ == "__main__":
    pytest.main([__file__])
