"""
Tests for embedding pipeline
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.embed import (
    EmbeddingConfig,
    EmbeddingModel,
    EmbeddingPipeline,
    FAISSIndex,
    create_embeddings_from_chunks_file,
    load_embedding_pipeline,
)
from src.ingest import ChunkMetadata, DocumentChunk


class TestEmbeddingConfig:
    """Test embedding configuration"""

    def test_init(self):
        """Test configuration initialization"""
        config = EmbeddingConfig(
            model_name="test-model",
            normalize_embeddings=True,
            device="cpu",
            similarity_threshold=0.8,
            top_k=10,
        )

        assert config.model_name == "test-model"
        assert config.normalize_embeddings is True
        assert config.device == "cpu"
        assert config.similarity_threshold == 0.8
        assert config.top_k == 10


class TestEmbeddingModel:
    """Test embedding model functionality"""

    @patch("src.embed.SentenceTransformer")
    def test_init(self, mock_sentence_transformer):
        """Test embedding model initialization"""
        config = EmbeddingConfig(
            model_name="all-MiniLM-L6-v2",
            normalize_embeddings=True,
            device="cpu",
            similarity_threshold=0.7,
            top_k=5,
        )

        mock_model = MagicMock()
        mock_sentence_transformer.return_value = mock_model

        embedding_model = EmbeddingModel(config)

        assert embedding_model.config == config
        mock_sentence_transformer.assert_called_with("all-MiniLM-L6-v2", device="cpu")

    @patch("src.embed.SentenceTransformer")
    def test_generate_embeddings(self, mock_sentence_transformer):
        """Test embedding generation"""
        config = EmbeddingConfig(
            model_name="all-MiniLM-L6-v2",
            normalize_embeddings=True,
            device="cpu",
            similarity_threshold=0.7,
            top_k=5,
        )

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_sentence_transformer.return_value = mock_model

        embedding_model = EmbeddingModel(config)

        texts = ["Hello world", "Test text"]
        embeddings = embedding_model.generate_embeddings(texts)

        assert embeddings.shape == (2, 3)
        mock_model.encode.assert_called_with(
            texts, normalize_embeddings=True, show_progress_bar=False
        )

    @patch("src.embed.SentenceTransformer")
    def test_generate_embeddings_empty(self, mock_sentence_transformer):
        """Test embedding generation with empty list"""
        config = EmbeddingConfig(
            model_name="all-MiniLM-L6-v2",
            normalize_embeddings=True,
            device="cpu",
            similarity_threshold=0.7,
            top_k=5,
        )

        mock_model = MagicMock()
        mock_sentence_transformer.return_value = mock_model

        embedding_model = EmbeddingModel(config)

        embeddings = embedding_model.generate_embeddings([])

        assert embeddings.shape == (0,)
        mock_model.encode.assert_not_called()

    @patch("src.embed.SentenceTransformer")
    def test_generate_single_embedding(self, mock_sentence_transformer):
        """Test single embedding generation"""
        config = EmbeddingConfig(
            model_name="all-MiniLM-L6-v2",
            normalize_embeddings=True,
            device="cpu",
            similarity_threshold=0.7,
            top_k=5,
        )

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_sentence_transformer.return_value = mock_model

        embedding_model = EmbeddingModel(config)

        embedding = embedding_model.generate_single_embedding("Hello world")

        assert embedding.shape == (3,)
        mock_model.encode.assert_called_with(
            ["Hello world"], normalize_embeddings=True, show_progress_bar=False
        )


class TestFAISSIndex:
    """Test FAISS index functionality"""

    def test_init(self):
        """Test FAISS index initialization"""
        index = FAISSIndex(dimension=384, index_type="IndexFlatIP")

        assert index.dimension == 384
        assert index.index_type == "IndexFlatIP"
        assert index.index is not None
        assert len(index.chunk_metadata) == 0

    def test_init_invalid_type(self):
        """Test initialization with invalid index type"""
        with pytest.raises(ValueError, match="Unsupported index type"):
            FAISSIndex(dimension=384, index_type="InvalidType")

    def test_add_embeddings(self):
        """Test adding embeddings to index"""
        index = FAISSIndex(dimension=3)

        embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        metadata = [
            ChunkMetadata(
                file_name="test1.pdf",
                page_number=1,
                chunk_index=0,
                chunk_start=0,
                chunk_end=100,
                chunk_size=100,
                text_length=1000,
            ),
            ChunkMetadata(
                file_name="test2.pdf",
                page_number=1,
                chunk_index=0,
                chunk_start=0,
                chunk_end=100,
                chunk_size=100,
                text_length=1000,
            ),
        ]

        index.add_embeddings(embeddings, metadata)

        assert index.get_total_embeddings() == 2
        assert len(index.chunk_metadata) == 2

    def test_add_embeddings_mismatch(self):
        """Test adding embeddings with mismatched metadata"""
        index = FAISSIndex(dimension=3)

        embeddings = np.array([[0.1, 0.2, 0.3]])
        metadata = [
            ChunkMetadata(
                file_name="test1.pdf",
                page_number=1,
                chunk_index=0,
                chunk_start=0,
                chunk_end=100,
                chunk_size=100,
                text_length=1000,
            ),
            ChunkMetadata(
                file_name="test2.pdf",
                page_number=1,
                chunk_index=0,
                chunk_start=0,
                chunk_end=100,
                chunk_size=100,
                text_length=1000,
            ),
        ]

        with pytest.raises(ValueError, match="Number of embeddings must match"):
            index.add_embeddings(embeddings, metadata)

    def test_search_empty_index(self):
        """Test search on empty index"""
        index = FAISSIndex(dimension=3)
        query_embedding = np.array([0.1, 0.2, 0.3])

        distances, indices = index.search(query_embedding, 5)

        assert len(distances) == 0
        assert len(indices) == 0

    def test_search_with_data(self):
        """Test search with data in index"""
        index = FAISSIndex(dimension=3)

        # Add some embeddings
        embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        metadata = [
            ChunkMetadata(
                file_name=f"test{i}.pdf",
                page_number=1,
                chunk_index=0,
                chunk_start=0,
                chunk_end=100,
                chunk_size=100,
                text_length=1000,
            )
            for i in range(3)
        ]

        index.add_embeddings(embeddings, metadata)

        # Search
        query_embedding = np.array([0.1, 0.2, 0.3])
        distances, indices = index.search(query_embedding, 2)

        assert len(distances) == 2
        assert len(indices) == 2
        assert all(0 <= idx < 3 for idx in indices)

    def test_get_chunk_by_index(self):
        """Test getting chunk metadata by index"""
        index = FAISSIndex(dimension=3)

        metadata = ChunkMetadata(
            file_name="test.pdf",
            page_number=1,
            chunk_index=0,
            chunk_start=0,
            chunk_end=100,
            chunk_size=100,
            text_length=1000,
        )

        index.chunk_metadata.append(metadata)

        # Valid index
        result = index.get_chunk_by_index(0)
        assert result == metadata

        # Invalid index
        result = index.get_chunk_by_index(1)
        assert result is None

    def test_save_and_load_index(self):
        """Test saving and loading index"""
        index = FAISSIndex(dimension=3)

        # Add some data
        embeddings = np.array([[0.1, 0.2, 0.3]])
        metadata = [
            ChunkMetadata(
                file_name="test.pdf",
                page_number=1,
                chunk_index=0,
                chunk_start=0,
                chunk_end=100,
                chunk_size=100,
                text_length=1000,
            )
        ]

        index.add_embeddings(embeddings, metadata)

        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = Path(temp_dir)

            # Save index
            index.save_index(index_path)

            # Verify files were created
            assert (index_path / "faiss.index").exists()
            assert (index_path / "chunk_metadata.json").exists()
            assert (index_path / "index_info.json").exists()

            # Create new index and load
            new_index = FAISSIndex(dimension=3)
            new_index.load_index(index_path)

            # Verify data was loaded correctly
            assert new_index.get_total_embeddings() == 1
            assert len(new_index.chunk_metadata) == 1
            assert new_index.chunk_metadata[0].file_name == "test.pdf"


class TestEmbeddingPipeline:
    """Test embedding pipeline functionality"""

    def test_init(self):
        """Test pipeline initialization"""
        config = {
            "embedding": {
                "model_name": "all-MiniLM-L6-v2",
                "normalize_embeddings": True,
                "device": "cpu",
                "similarity_threshold": 0.7,
                "top_k": 5,
            }
        }

        with patch("src.embed.EmbeddingModel"):
            pipeline = EmbeddingPipeline(config)

            assert pipeline.config == config
            assert pipeline.faiss_index is None
            assert pipeline.index_loaded is False

    @patch("src.embed.EmbeddingModel")
    def test_create_embeddings_from_chunks(self, mock_embedding_model):
        """Test creating embeddings from chunks"""
        config = {
            "embedding": {
                "model_name": "all-MiniLM-L6-v2",
                "normalize_embeddings": True,
                "device": "cpu",
                "similarity_threshold": 0.7,
                "top_k": 5,
            }
        }

        # Mock embedding model
        mock_model = MagicMock()
        mock_model.generate_embeddings.return_value = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        )
        mock_embedding_model.return_value = mock_model

        pipeline = EmbeddingPipeline(config)

        # Create test chunks
        chunks = [
            DocumentChunk(
                text="Hello world",
                metadata=ChunkMetadata(
                    file_name="test1.pdf",
                    page_number=1,
                    chunk_index=0,
                    chunk_start=0,
                    chunk_end=100,
                    chunk_size=100,
                    text_length=1000,
                ),
            ),
            DocumentChunk(
                text="Test text",
                metadata=ChunkMetadata(
                    file_name="test2.pdf",
                    page_number=1,
                    chunk_index=0,
                    chunk_start=0,
                    chunk_end=100,
                    chunk_size=100,
                    text_length=1000,
                ),
            ),
        ]

        pipeline.create_embeddings_from_chunks(chunks)

        assert pipeline.faiss_index is not None
        assert pipeline.faiss_index.get_total_embeddings() == 2
        assert len(pipeline.faiss_index.chunk_metadata) == 2

    @patch("src.embed.EmbeddingModel")
    def test_create_embeddings_empty_chunks(self, mock_embedding_model):
        """Test creating embeddings with empty chunks"""
        config = {
            "embedding": {
                "model_name": "all-MiniLM-L6-v2",
                "normalize_embeddings": True,
                "device": "cpu",
                "similarity_threshold": 0.7,
                "top_k": 5,
            }
        }

        mock_model = MagicMock()
        mock_embedding_model.return_value = mock_model

        pipeline = EmbeddingPipeline(config)

        pipeline.create_embeddings_from_chunks([])

        assert pipeline.faiss_index is None
        mock_model.generate_embeddings.assert_not_called()

    @patch("src.embed.EmbeddingModel")
    def test_save_and_load_index(self, mock_embedding_model):
        """Test saving and loading index"""
        config = {
            "embedding": {
                "model_name": "all-MiniLM-L6-v2",
                "normalize_embeddings": True,
                "device": "cpu",
                "similarity_threshold": 0.7,
                "top_k": 5,
            }
        }

        # Mock embedding model with proper config
        mock_model = MagicMock()
        mock_model.config = EmbeddingConfig(
            model_name="all-MiniLM-L6-v2",
            normalize_embeddings=True,
            device="cpu",
            similarity_threshold=0.7,
            top_k=5,
        )
        mock_model.generate_embeddings.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_embedding_model.return_value = mock_model

        pipeline = EmbeddingPipeline(config)

        # Create test chunks
        chunks = [
            DocumentChunk(
                text="Hello world",
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
        ]

        pipeline.create_embeddings_from_chunks(chunks)

        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = Path(temp_dir)

            # Save index
            pipeline.save_index(index_path)

            # Verify files were created
            assert (index_path / "faiss.index").exists()
            assert (index_path / "chunk_metadata.json").exists()
            assert (index_path / "index_info.json").exists()
            assert (index_path / "model_info.json").exists()

            # Create new pipeline and load
            new_pipeline = EmbeddingPipeline(config)
            new_pipeline.load_index(index_path)

            # Verify data was loaded correctly
            assert new_pipeline.index_loaded is True
            assert new_pipeline.faiss_index.get_total_embeddings() == 1

    @patch("src.embed.EmbeddingModel")
    def test_search_similar_chunks(self, mock_embedding_model):
        """Test searching for similar chunks"""
        config = {
            "embedding": {
                "model_name": "all-MiniLM-L6-v2",
                "normalize_embeddings": True,
                "device": "cpu",
                "similarity_threshold": 0.5,
                "top_k": 5,
            }
        }

        # Mock embedding model with proper config
        mock_model = MagicMock()
        mock_model.config = EmbeddingConfig(
            model_name="all-MiniLM-L6-v2",
            normalize_embeddings=True,
            device="cpu",
            similarity_threshold=0.5,
            top_k=5,
        )
        mock_model.generate_single_embedding.return_value = np.array([0.1, 0.2, 0.3])
        mock_embedding_model.return_value = mock_model

        pipeline = EmbeddingPipeline(config)

        # Create a mock FAISS index
        pipeline.faiss_index = MagicMock()
        pipeline.faiss_index.index_type = "IndexFlatIP"
        pipeline.faiss_index.search.return_value = (
            np.array([0.8, 0.6]),
            np.array([0, 1]),
        )
        pipeline.faiss_index.get_chunk_by_index.side_effect = lambda idx: ChunkMetadata(
            file_name=f"test{idx}.pdf",
            page_number=1,
            chunk_index=0,
            chunk_start=0,
            chunk_end=100,
            chunk_size=100,
            text_length=1000,
        )

        results = pipeline.search_similar_chunks("test query")

        assert len(results) == 2
        assert all(isinstance(result[0], DocumentChunk) for result in results)
        assert all(isinstance(result[1], float) for result in results)
        assert all(result[1] >= 0.5 for result in results)  # Above threshold

    @patch("src.embed.EmbeddingModel")
    def test_search_no_index(self, mock_embedding_model):
        """Test search without loaded index"""
        config = {
            "embedding": {
                "model_name": "all-MiniLM-L6-v2",
                "normalize_embeddings": True,
                "device": "cpu",
                "similarity_threshold": 0.7,
                "top_k": 5,
            }
        }

        mock_model = MagicMock()
        mock_embedding_model.return_value = mock_model

        pipeline = EmbeddingPipeline(config)

        with pytest.raises(ValueError, match="No index loaded"):
            pipeline.search_similar_chunks("test query")

    @patch("src.embed.EmbeddingModel")
    def test_get_index_stats(self, mock_embedding_model):
        """Test getting index statistics"""
        config = {
            "embedding": {
                "model_name": "all-MiniLM-L6-v2",
                "normalize_embeddings": True,
                "device": "cpu",
                "similarity_threshold": 0.7,
                "top_k": 5,
            }
        }

        # Mock embedding model with proper config
        mock_model = MagicMock()
        mock_model.config = EmbeddingConfig(
            model_name="all-MiniLM-L6-v2",
            normalize_embeddings=True,
            device="cpu",
            similarity_threshold=0.7,
            top_k=5,
        )
        mock_embedding_model.return_value = mock_model

        pipeline = EmbeddingPipeline(config)

        # Test without index
        stats = pipeline.get_index_stats()
        assert "error" in stats

        # Test with index
        pipeline.faiss_index = MagicMock()
        pipeline.faiss_index.get_total_embeddings.return_value = 10
        pipeline.faiss_index.dimension = 384
        pipeline.faiss_index.index_type = "IndexFlatIP"

        stats = pipeline.get_index_stats()
        assert stats["total_embeddings"] == 10
        assert stats["dimension"] == 384
        assert stats["index_type"] == "IndexFlatIP"
        assert stats["model_name"] == "all-MiniLM-L6-v2"


class TestEmbeddingFunctions:
    """Test embedding utility functions"""

    @patch("src.embed.EmbeddingPipeline")
    def test_create_embeddings_from_chunks_file(self, mock_pipeline_class):
        """Test creating embeddings from chunks file"""
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline

        config = {"embedding": {"model_name": "test-model"}}

        with tempfile.TemporaryDirectory() as temp_dir:
            chunks_file = Path(temp_dir) / "chunks.json"
            output_path = Path(temp_dir) / "index"

            # Create test chunks file
            chunks_data = [
                {
                    "text": "Hello world",
                    "metadata": {
                        "file_name": "test.pdf",
                        "page_number": 1,
                        "chunk_index": 0,
                        "chunk_start": 0,
                        "chunk_end": 100,
                        "chunk_size": 100,
                        "text_length": 1000,
                    },
                }
            ]

            with open(chunks_file, "w") as f:
                json.dump(chunks_data, f)

            create_embeddings_from_chunks_file(chunks_file, config, output_path)

            mock_pipeline.create_embeddings_from_chunks.assert_called_once()
            mock_pipeline.save_index.assert_called_once_with(output_path)

    @patch("src.embed.EmbeddingPipeline")
    def test_load_embedding_pipeline(self, mock_pipeline_class):
        """Test loading embedding pipeline"""
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline

        config = {"embedding": {"model_name": "test-model"}}

        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = Path(temp_dir)

            result = load_embedding_pipeline(config, index_path)

            assert result == mock_pipeline
            mock_pipeline.load_index.assert_called_once_with(index_path)


if __name__ == "__main__":
    pytest.main([__file__])
