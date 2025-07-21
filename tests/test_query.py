"""
Tests for query engine
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ingest import ChunkMetadata, DocumentChunk
from src.query import (
    QueryEngine,
    QueryProcessor,
    QueryResult,
    format_query_output,
    process_query,
)


class TestQueryResult:
    """Test QueryResult dataclass"""

    def test_init(self):
        """Test QueryResult initialization"""
        chunks = [
            DocumentChunk(
                text="Test chunk 1",
                metadata=ChunkMetadata(
                    file_name="test.pdf",
                    page_number=1,
                    chunk_index=0,
                    chunk_start=0,
                    chunk_end=100,
                    chunk_size=100,
                    text_length=1000
                )
            )
        ]

        result = QueryResult(
            query="test query",
            chunks=chunks,
            similarities=[0.85],
            total_chunks_searched=100,
            search_time_ms=50.0
        )

        assert result.query == "test query"
        assert len(result.chunks) == 1
        assert result.similarities == [0.85]
        assert result.total_chunks_searched == 100
        assert result.search_time_ms == 50.0


class TestQueryEngine:
    """Test query engine functionality"""

    @patch('src.query.load_embedding_pipeline')
    def test_init(self, mock_load_pipeline):
        """Test query engine initialization"""
        config = {"embedding": {"model_name": "test-model"}}
        mock_pipeline = MagicMock()
        mock_load_pipeline.return_value = mock_pipeline

        engine = QueryEngine(config)

        assert engine.config == config
        assert engine.embedding_pipeline == mock_pipeline
        mock_load_pipeline.assert_called_once()

    @patch('src.query.load_embedding_pipeline')
    def test_init_with_index_path(self, mock_load_pipeline):
        """Test initialization with custom index path"""
        config = {"embedding": {"model_name": "test-model"}}
        index_path = Path("/custom/index/path")
        mock_pipeline = MagicMock()
        mock_load_pipeline.return_value = mock_pipeline

        engine = QueryEngine(config, index_path)

        assert engine.index_path == index_path
        mock_load_pipeline.assert_called_once_with(config, index_path)

    @patch('src.query.load_embedding_pipeline')
    def test_search_empty_query(self, mock_load_pipeline):
        """Test search with empty query"""
        config = {"embedding": {"model_name": "test-model"}}
        mock_pipeline = MagicMock()
        mock_load_pipeline.return_value = mock_pipeline

        engine = QueryEngine(config)

        result = engine.search("")

        assert result.query == ""
        assert len(result.chunks) == 0
        assert len(result.similarities) == 0
        assert result.total_chunks_searched == 0
        assert result.search_time_ms == 0.0

    @patch('src.query.load_embedding_pipeline')
    def test_search_success(self, mock_load_pipeline):
        """Test successful search"""
        config = {
            "embedding": {
                "model_name": "test-model",
                "top_k": 5,
                "similarity_threshold": 0.7
            }
        }

        # Mock embedding pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.embedding_model.config.top_k = 5
        mock_pipeline.embedding_model.config.similarity_threshold = 0.7
        mock_pipeline.faiss_index.get_total_embeddings.return_value = 100

        # Mock search results
        chunks = [
            DocumentChunk(
                text="Test chunk 1",
                metadata=ChunkMetadata(
                    file_name="test1.pdf",
                    page_number=1,
                    chunk_index=0,
                    chunk_start=0,
                    chunk_end=100,
                    chunk_size=100,
                    text_length=1000
                )
            ),
            DocumentChunk(
                text="Test chunk 2",
                metadata=ChunkMetadata(
                    file_name="test2.pdf",
                    page_number=1,
                    chunk_index=0,
                    chunk_start=0,
                    chunk_end=100,
                    chunk_size=100,
                    text_length=1000
                )
            )
        ]

        mock_pipeline.search_similar_chunks.return_value = [
            (chunks[0], 0.85),
            (chunks[1], 0.75)
        ]

        mock_load_pipeline.return_value = mock_pipeline

        engine = QueryEngine(config)

        result = engine.search("test query")

        assert result.query == "test query"
        assert len(result.chunks) == 2
        assert result.similarities == [0.85, 0.75]
        assert result.total_chunks_searched == 100
        assert result.search_time_ms >= 0  # Allow 0 for mocked functions

    @patch('src.query.load_embedding_pipeline')
    def test_search_with_custom_params(self, mock_load_pipeline):
        """Test search with custom parameters"""
        config = {
            "embedding": {
                "model_name": "test-model",
                "top_k": 5,
                "similarity_threshold": 0.7
            }
        }

        mock_pipeline = MagicMock()
        mock_pipeline.embedding_model.config.top_k = 5
        mock_pipeline.embedding_model.config.similarity_threshold = 0.7
        mock_pipeline.faiss_index.get_total_embeddings.return_value = 100
        mock_pipeline.search_similar_chunks.return_value = []

        mock_load_pipeline.return_value = mock_pipeline

        engine = QueryEngine(config)

        result = engine.search("test query", top_k=10, similarity_threshold=0.8)

        # Verify custom parameters were used
        mock_pipeline.search_similar_chunks.assert_called_with("test query", 10)

    @patch('src.query.load_embedding_pipeline')
    def test_get_index_stats(self, mock_load_pipeline):
        """Test getting index statistics"""
        config = {"embedding": {"model_name": "test-model"}}
        mock_pipeline = MagicMock()
        mock_pipeline.get_index_stats.return_value = {
            "total_embeddings": 100,
            "dimension": 384,
            "model_name": "test-model"
        }
        mock_load_pipeline.return_value = mock_pipeline

        engine = QueryEngine(config)

        stats = engine.get_index_stats()

        assert stats["total_embeddings"] == 100
        assert stats["dimension"] == 384
        assert stats["model_name"] == "test-model"

    @patch('src.query.load_embedding_pipeline')
    def test_validate_index_success(self, mock_load_pipeline):
        """Test successful index validation"""
        config = {"embedding": {"model_name": "test-model"}}
        mock_pipeline = MagicMock()
        mock_pipeline.get_index_stats.return_value = {
            "total_embeddings": 100,
            "dimension": 384
        }
        mock_pipeline.search_similar_chunks.return_value = []
        mock_load_pipeline.return_value = mock_pipeline

        engine = QueryEngine(config)

        assert engine.validate_index() is True

    @patch('src.query.load_embedding_pipeline')
    def test_validate_index_no_embeddings(self, mock_load_pipeline):
        """Test index validation with no embeddings"""
        config = {"embedding": {"model_name": "test-model"}}
        mock_pipeline = MagicMock()
        mock_pipeline.get_index_stats.return_value = {
            "total_embeddings": 0,
            "dimension": 384
        }
        mock_load_pipeline.return_value = mock_pipeline

        engine = QueryEngine(config)

        assert engine.validate_index() is False

    @patch('src.query.load_embedding_pipeline')
    def test_validate_index_error(self, mock_load_pipeline):
        """Test index validation with error"""
        config = {"embedding": {"model_name": "test-model"}}
        mock_pipeline = MagicMock()
        mock_pipeline.get_index_stats.return_value = {"error": "Index not found"}
        mock_load_pipeline.return_value = mock_pipeline

        engine = QueryEngine(config)

        assert engine.validate_index() is False


class TestQueryProcessor:
    """Test query processor functionality"""

    @patch('src.query.QueryEngine')
    def test_init(self, mock_query_engine):
        """Test query processor initialization"""
        config = {"embedding": {"model_name": "test-model"}}
        mock_engine = MagicMock()
        mock_query_engine.return_value = mock_engine

        processor = QueryProcessor(config)

        assert processor.config == config
        assert processor.query_engine == mock_engine
        assert processor.chunk_texts == {}

    @patch('src.query.QueryEngine')
    def test_load_chunk_texts(self, mock_query_engine):
        """Test loading chunk texts from chunks.json"""
        config = {"embedding": {"model_name": "test-model"}}
        mock_engine = MagicMock()
        mock_engine.index_path = Path("/test/index")
        mock_query_engine.return_value = mock_engine

        # Mock chunks.json content
        chunks_data = [
            {
                "text": "Test chunk 1",
                "metadata": {
                    "file_name": "test1.pdf",
                    "page_number": 1,
                    "chunk_index": 0,
                    "chunk_start": 0,
                    "chunk_end": 100,
                    "chunk_size": 100,
                    "text_length": 1000
                }
            },
            {
                "text": "Test chunk 2",
                "metadata": {
                    "file_name": "test2.pdf",
                    "page_number": 1,
                    "chunk_index": 0,
                    "chunk_start": 0,
                    "chunk_end": 100,
                    "chunk_size": 100,
                    "text_length": 1000
                }
            }
        ]

        with patch('builtins.open', mock_open(read_data=json.dumps(chunks_data))):
            with patch('pathlib.Path.exists', return_value=True):
                processor = QueryProcessor(config)

                assert len(processor.chunk_texts) == 2
                assert "test1.pdf_1_0" in processor.chunk_texts
                assert "test2.pdf_1_0" in processor.chunk_texts
                assert processor.chunk_texts["test1.pdf_1_0"] == "Test chunk 1"

    @patch('src.query.QueryEngine')
    def test_process_query(self, mock_query_engine):
        """Test query processing"""
        config = {"embedding": {"model_name": "test-model"}}
        mock_engine = MagicMock()
        mock_engine.index_path = Path("/test/index")

        # Mock search result
        chunks = [
            DocumentChunk(
                text="[Text not stored in index]",
                metadata=ChunkMetadata(
                    file_name="test1.pdf",
                    page_number=1,
                    chunk_index=0,
                    chunk_start=0,
                    chunk_end=100,
                    chunk_size=100,
                    text_length=1000
                )
            )
        ]

        search_result = QueryResult(
            query="test query",
            chunks=chunks,
            similarities=[0.85],
            total_chunks_searched=100,
            search_time_ms=50.0
        )

        mock_engine.search.return_value = search_result
        mock_query_engine.return_value = mock_engine

        # Mock chunk texts
        processor = QueryProcessor(config)
        processor.chunk_texts = {
            "test1.pdf_1_0": "Actual chunk text"
        }

        result = processor.process_query("test query")

        assert result.query == "test query"
        assert len(result.chunks) == 1
        assert result.chunks[0].text == "Actual chunk text"  # Enhanced with actual text
        assert result.similarities == [0.85]

    @patch('src.query.QueryEngine')
    def test_format_results(self, mock_query_engine):
        """Test result formatting"""
        config = {"embedding": {"model_name": "test-model"}}
        mock_engine = MagicMock()
        mock_query_engine.return_value = mock_engine

        processor = QueryProcessor(config)

        # Create test result
        chunks = [
            DocumentChunk(
                text="This is a test chunk with some content.",
                metadata=ChunkMetadata(
                    file_name="test.pdf",
                    page_number=1,
                    chunk_index=0,
                    chunk_start=0,
                    chunk_end=100,
                    chunk_size=100,
                    text_length=1000
                )
            )
        ]

        result = QueryResult(
            query="test query",
            chunks=chunks,
            similarities=[0.85],
            total_chunks_searched=100,
            search_time_ms=50.0
        )

        formatted = processor.format_results(result, include_metadata=True)

        assert "Query: 'test query'" in formatted
        assert "Found 1 relevant chunks" in formatted
        assert "File: test.pdf" in formatted
        assert "Page: 1" in formatted
        assert "This is a test chunk" in formatted

    @patch('src.query.QueryEngine')
    def test_format_results_no_chunks(self, mock_query_engine):
        """Test formatting with no results"""
        config = {"embedding": {"model_name": "test-model"}}
        mock_engine = MagicMock()
        mock_query_engine.return_value = mock_engine

        processor = QueryProcessor(config)

        result = QueryResult(
            query="test query",
            chunks=[],
            similarities=[],
            total_chunks_searched=100,
            search_time_ms=50.0
        )

        formatted = processor.format_results(result)

        assert "No relevant chunks found" in formatted
        assert "test query" in formatted

    @patch('src.query.QueryEngine')
    def test_get_relevant_context(self, mock_query_engine):
        """Test getting relevant context for LLM"""
        config = {"embedding": {"model_name": "test-model"}}
        mock_engine = MagicMock()
        mock_query_engine.return_value = mock_engine

        processor = QueryProcessor(config)

        # Create test result
        chunks = [
            DocumentChunk(
                text="This is the first chunk with some content.",
                metadata=ChunkMetadata(
                    file_name="test1.pdf",
                    page_number=1,
                    chunk_index=0,
                    chunk_start=0,
                    chunk_end=100,
                    chunk_size=100,
                    text_length=1000
                )
            ),
            DocumentChunk(
                text="This is the second chunk with different content.",
                metadata=ChunkMetadata(
                    file_name="test2.pdf",
                    page_number=2,
                    chunk_index=0,
                    chunk_start=0,
                    chunk_end=100,
                    chunk_size=100,
                    text_length=1000
                )
            )
        ]

        result = QueryResult(
            query="test query",
            chunks=chunks,
            similarities=[0.85, 0.75],
            total_chunks_searched=100,
            search_time_ms=50.0
        )

        context = processor.get_relevant_context(result, max_chars=100)

        assert "Document: test1.pdf" in context
        assert "Page: 1" in context
        assert "This is the first chunk" in context
        assert len(context) <= 100  # Respect max_chars limit

    @patch('src.query.QueryEngine')
    def test_get_relevant_context_no_chunks(self, mock_query_engine):
        """Test getting context with no chunks"""
        config = {"embedding": {"model_name": "test-model"}}
        mock_engine = MagicMock()
        mock_query_engine.return_value = mock_engine

        processor = QueryProcessor(config)

        result = QueryResult(
            query="test query",
            chunks=[],
            similarities=[],
            total_chunks_searched=100,
            search_time_ms=50.0
        )

        context = processor.get_relevant_context(result)

        assert context == "No relevant information found."


class TestQueryFunctions:
    """Test query utility functions"""

    @patch('src.query.QueryProcessor')
    def test_process_query_success(self, mock_processor_class):
        """Test successful query processing"""
        config = {"embedding": {"model_name": "test-model"}}
        args = MagicMock()
        args.top_k = 10
        args.similarity_threshold = 0.8

        mock_processor = MagicMock()
        mock_processor.query_engine.validate_index.return_value = True

        chunks = [
            DocumentChunk(
                text="Test chunk",
                metadata=ChunkMetadata(
                    file_name="test.pdf",
                    page_number=1,
                    chunk_index=0,
                    chunk_start=0,
                    chunk_end=100,
                    chunk_size=100,
                    text_length=1000
                )
            )
        ]

        mock_result = QueryResult(
            query="test query",
            chunks=chunks,
            similarities=[0.85],
            total_chunks_searched=100,
            search_time_ms=50.0
        )

        mock_processor.process_query.return_value = mock_result
        mock_processor_class.return_value = mock_processor

        result = process_query("test query", config, args)

        assert result == mock_result
        mock_processor.process_query.assert_called_once()

    @patch('src.query.QueryProcessor')
    def test_process_query_validation_failed(self, mock_processor_class):
        """Test query processing with validation failure"""
        config = {"embedding": {"model_name": "test-model"}}
        args = MagicMock()

        mock_processor = MagicMock()
        mock_processor.query_engine.validate_index.return_value = False
        mock_processor_class.return_value = mock_processor

        with pytest.raises(ValueError, match="Index validation failed"):
            process_query("test query", config, args)

    def test_format_query_output_verbose(self):
        """Test verbose query output formatting"""
        chunks = [
            DocumentChunk(
                text="This is a test chunk with some content.",
                metadata=ChunkMetadata(
                    file_name="test.pdf",
                    page_number=1,
                    chunk_index=0,
                    chunk_start=0,
                    chunk_end=100,
                    chunk_size=100,
                    text_length=1000
                )
            )
        ]

        result = QueryResult(
            query="test query",
            chunks=chunks,
            similarities=[0.85],
            total_chunks_searched=100,
            search_time_ms=50.0
        )

        output = format_query_output(result, verbose=True)

        assert "Query: 'test query'" in output
        assert "Found 1 relevant chunks" in output
        assert "File: test.pdf" in output
        assert "Page: 1" in output

    def test_format_query_output_simple(self):
        """Test simple query output formatting"""
        chunks = [
            DocumentChunk(
                text="This is a test chunk with some content.",
                metadata=ChunkMetadata(
                    file_name="test.pdf",
                    page_number=1,
                    chunk_index=0,
                    chunk_start=0,
                    chunk_end=100,
                    chunk_size=100,
                    text_length=1000
                )
            )
        ]

        result = QueryResult(
            query="test query",
            chunks=chunks,
            similarities=[0.85],
            total_chunks_searched=100,
            search_time_ms=50.0
        )

        output = format_query_output(result, verbose=False)

        assert "Found 1 relevant chunks" in output
        assert "test.pdf" in output
        assert "p.1" in output
        assert "similarity: 0.850" in output

    def test_format_query_output_no_results(self):
        """Test output formatting with no results"""
        result = QueryResult(
            query="test query",
            chunks=[],
            similarities=[],
            total_chunks_searched=100,
            search_time_ms=50.0
        )

        output = format_query_output(result, verbose=False)

        assert "No relevant information found" in output
        assert "test query" in output


if __name__ == "__main__":
    pytest.main([__file__])
