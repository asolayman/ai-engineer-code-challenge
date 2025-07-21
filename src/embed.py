"""
Embedding Pipeline

Handles vector embedding generation for document chunks and FAISS index management.
Supports local embedding models and efficient similarity search.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.ingest import ChunkMetadata, DocumentChunk

# Import utility functions
from .utils import (
    get_logger,
    log_memory_usage,
    log_performance,
)

logger = get_logger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    model_name: str
    normalize_embeddings: bool
    device: str
    similarity_threshold: float
    top_k: int


class EmbeddingModel:
    """Handles embedding model loading and text embedding generation."""

    def __init__(self, config: EmbeddingConfig):
        """
        Initialize embedding model.
        
        Args:
            config: Embedding configuration
        """
        self.config = config
        self.model = None
        self._load_model()
        logger.info(f"Initialized embedding model: {config.model_name}")

    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        try:
            self.model = SentenceTransformer(
                self.config.model_name,
                device=self.config.device
            )
            logger.info(f"Loaded model {self.config.model_name} on {self.config.device}")
        except Exception as e:
            logger.error(f"Failed to load model {self.config.model_name}: {e}")
            raise

    def generate_embeddings(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            numpy array of embeddings
        """
        if not texts:
            return np.array([])

        try:
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=self.config.normalize_embeddings,
                show_progress_bar=False
            )
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def generate_single_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string to embed
            
        Returns:
            numpy array of embedding
        """
        return self.generate_embeddings([text])[0]


class FAISSIndex:
    """Handles FAISS index creation and management."""

    def __init__(self, dimension: int, index_type: str = "IndexFlatIP"):
        """
        Initialize FAISS index.
        
        Args:
            dimension: Dimension of embeddings
            index_type: Type of FAISS index to use
        """
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.chunk_metadata = []
        self._create_index()
        logger.info(f"Initialized FAISS index: {index_type} with dimension {dimension}")

    def _create_index(self) -> None:
        """Create FAISS index based on type."""
        if self.index_type == "IndexFlatIP":
            self.index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "IndexFlatL2":
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "IndexIVFFlat":
            # For IVF, we need to train on some data first
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")

    def add_embeddings(self, embeddings: np.ndarray, chunk_metadata: list[ChunkMetadata]) -> None:
        """
        Add embeddings to the index.
        
        Args:
            embeddings: numpy array of embeddings
            chunk_metadata: List of chunk metadata corresponding to embeddings
        """
        if len(embeddings) != len(chunk_metadata):
            raise ValueError("Number of embeddings must match number of metadata entries")

        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))

        # Store metadata
        self.chunk_metadata.extend(chunk_metadata)

        logger.info(f"Added {len(embeddings)} embeddings to index")

    def search(self, query_embedding: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: Query embedding
            k: Number of results to return
            
        Returns:
            Tuple of (distances, indices)
        """
        if self.index.ntotal == 0:
            return np.array([]), np.array([])

        # Reshape query embedding for FAISS
        query_embedding = query_embedding.reshape(1, -1).astype('float32')

        # Search
        distances, indices = self.index.search(query_embedding, min(k, self.index.ntotal))

        return distances[0], indices[0]

    def get_chunk_by_index(self, index: int) -> ChunkMetadata | None:
        """
        Get chunk metadata by index.
        
        Args:
            index: Index in the metadata list
            
        Returns:
            Chunk metadata or None if index is invalid
        """
        if 0 <= index < len(self.chunk_metadata):
            return self.chunk_metadata[index]
        return None

    def get_total_embeddings(self) -> int:
        """Get total number of embeddings in index."""
        return self.index.ntotal

    def save_index(self, index_path: Path) -> None:
        """
        Save FAISS index and metadata to disk.
        
        Args:
            index_path: Path to save index
        """
        index_path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(index_path / "faiss.index"))

        # Save metadata
        metadata_file = index_path / "chunk_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            metadata_list = [asdict(meta) for meta in self.chunk_metadata]
            json.dump(metadata_list, f, indent=2, ensure_ascii=False)

        # Save index info
        info = {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "total_embeddings": self.get_total_embeddings(),
            "model_name": "sentence-transformers"  # This will be updated by EmbeddingPipeline
        }

        info_file = index_path / "index_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2)

        logger.info(f"Saved index to {index_path}")

    def load_index(self, index_path: Path) -> None:
        """
        Load FAISS index and metadata from disk.
        
        Args:
            index_path: Path to load index from
        """
        # Load FAISS index
        faiss_index_file = index_path / "faiss.index"
        if not faiss_index_file.exists():
            raise FileNotFoundError(f"FAISS index file not found: {faiss_index_file}")

        self.index = faiss.read_index(str(faiss_index_file))

        # Load metadata
        metadata_file = index_path / "chunk_metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        with open(metadata_file, encoding='utf-8') as f:
            metadata_list = json.load(f)
            self.chunk_metadata = [ChunkMetadata(**meta) for meta in metadata_list]

        # Load index info
        info_file = index_path / "index_info.json"
        if info_file.exists():
            with open(info_file, encoding='utf-8') as f:
                info = json.load(f)
                self.dimension = info.get("dimension", self.dimension)
                self.index_type = info.get("index_type", self.index_type)

        logger.info(f"Loaded index from {index_path} with {len(self.chunk_metadata)} chunks")


class EmbeddingPipeline:
    """Main class for embedding generation and index management."""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize embedding pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config

        # Initialize embedding model
        embedding_config = EmbeddingConfig(
            model_name=config.get("embedding", {}).get("model_name", "all-MiniLM-L6-v2"),
            normalize_embeddings=config.get("embedding", {}).get("normalize_embeddings", True),
            device=config.get("embedding", {}).get("device", "cpu"),
            similarity_threshold=config.get("embedding", {}).get("similarity_threshold", 0.7),
            top_k=config.get("embedding", {}).get("top_k", 5)
        )

        self.embedding_model = EmbeddingModel(embedding_config)
        self.faiss_index = None
        self.index_loaded = False

        logger.info("Initialized EmbeddingPipeline")

    @log_performance
    def create_embeddings_from_chunks(self, chunks: list[DocumentChunk]) -> None:
        """
        Create embeddings from document chunks and build FAISS index.
        
        Args:
            chunks: List of document chunks
        """
        if not chunks:
            logger.warning("No chunks provided for embedding")
            return

        logger.info(f"Creating embeddings for {len(chunks)} chunks")
        log_memory_usage(logger, "Before embedding creation")

        # Extract texts and metadata
        texts = [chunk.text for chunk in chunks]
        metadata = [chunk.metadata for chunk in chunks]

        # Get batch size from config
        batch_size = self.config.get("system", {}).get("batch_size", 32)

        # Process embeddings in batches
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            logger.debug(f"Processing embedding batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")

            try:
                batch_embeddings = self.embedding_model.generate_embeddings(batch_texts)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Error encoding batch: {e}")
                # Add zero embeddings for failed batch
                zero_embeddings = [np.zeros(self.embedding_model.config.dimension)] * len(batch_texts)
                all_embeddings.extend(zero_embeddings)

        # Convert to numpy array
        embeddings = np.array(all_embeddings)

        # Create FAISS index
        dimension = embeddings.shape[1]
        self.faiss_index = FAISSIndex(dimension)

        # Add embeddings to index
        self.faiss_index.add_embeddings(embeddings, metadata)

        logger.info(f"Created embeddings with dimension {dimension}")
        log_memory_usage(logger, "After embedding creation")

    def save_index(self, index_path: Path) -> None:
        """
        Save the FAISS index and metadata.
        
        Args:
            index_path: Path to save index
        """
        if self.faiss_index is None:
            raise ValueError("No index to save. Run create_embeddings_from_chunks first.")

        self.faiss_index.save_index(index_path)

        # Save embedding model info
        model_info = {
            "model_name": self.embedding_model.config.model_name,
            "normalize_embeddings": self.embedding_model.config.normalize_embeddings,
            "device": self.embedding_model.config.device,
            "similarity_threshold": self.embedding_model.config.similarity_threshold,
            "top_k": self.embedding_model.config.top_k
        }

        model_info_file = index_path / "model_info.json"
        with open(model_info_file, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2)

    def load_index(self, index_path: Path) -> None:
        """
        Load the FAISS index and metadata.
        
        Args:
            index_path: Path to load index from
        """
        if not index_path.exists():
            raise FileNotFoundError(f"Index directory not found: {index_path}")

        # Load FAISS index
        self.faiss_index = FAISSIndex(384)  # Default dimension, will be updated
        self.faiss_index.load_index(index_path)

        # Load model info
        model_info_file = index_path / "model_info.json"
        if model_info_file.exists():
            with open(model_info_file, encoding='utf-8') as f:
                model_info = json.load(f)

                # Update embedding model config
                self.embedding_model.config.model_name = model_info.get("model_name", "all-MiniLM-L6-v2")
                self.embedding_model.config.normalize_embeddings = model_info.get("normalize_embeddings", True)
                self.embedding_model.config.device = model_info.get("device", "cpu")
                self.embedding_model.config.similarity_threshold = model_info.get("similarity_threshold", 0.7)
                self.embedding_model.config.top_k = model_info.get("top_k", 5)

        self.index_loaded = True
        logger.info(f"Loaded index with {self.faiss_index.get_total_embeddings()} embeddings")

    def search_similar_chunks(self, query: str, top_k: int | None = None) -> list[tuple[DocumentChunk, float]]:
        """
        Search for chunks similar to the query.
        
        Args:
            query: Query text
            top_k: Number of results to return (uses config default if None)
            
        Returns:
            List of (chunk, similarity_score) tuples
        """
        if self.faiss_index is None:
            raise ValueError("No index loaded. Load index first.")

        if top_k is None:
            top_k = self.embedding_model.config.top_k

        # Generate query embedding
        query_embedding = self.embedding_model.generate_single_embedding(query)

        # Search index
        distances, indices = self.faiss_index.search(query_embedding, top_k)

        # Debug logging
        logger.debug(f"FAISS search returned {len(distances)} results")
        logger.debug(f"Distances: {distances}")
        logger.debug(f"Indices: {indices}")

        # Convert to similarity scores (for IP index, higher is better)
        if self.faiss_index.index_type == "IndexFlatIP":
            similarities = distances
        else:
            # For L2 distance, convert to similarity (lower distance = higher similarity)
            similarities = 1.0 / (1.0 + distances)

        logger.debug(f"Similarities: {similarities}")

        # Return all results (threshold filtering will be done by query engine)
        results = []

        for idx, similarity in zip(indices, similarities, strict=False):
            metadata = self.faiss_index.get_chunk_by_index(idx)
            if metadata:
                # Reconstruct chunk (we don't store text in index to save space)
                # In a real implementation, you might want to store text or load from chunks.json
                chunk = DocumentChunk(text="[Text not stored in index]", metadata=metadata)
                results.append((chunk, float(similarity)))

        logger.debug(f"Returning {len(results)} results")
        return results

    def get_index_stats(self) -> dict[str, Any]:
        """
        Get statistics about the index.
        
        Returns:
            Dictionary with index statistics
        """
        if self.faiss_index is None:
            return {"error": "No index loaded"}

        return {
            "total_embeddings": self.faiss_index.get_total_embeddings(),
            "dimension": self.faiss_index.dimension,
            "index_type": self.faiss_index.index_type,
            "model_name": self.embedding_model.config.model_name,
            "similarity_threshold": self.embedding_model.config.similarity_threshold,
            "top_k": self.embedding_model.config.top_k
        }


def create_embeddings_from_chunks_file(chunks_file: Path, config: dict[str, Any], output_path: Path) -> None:
    """
    Create embeddings from a chunks.json file.
    
    Args:
        chunks_file: Path to chunks.json file
        config: Configuration dictionary
        output_path: Path to save index
    """
    # Load chunks
    with open(chunks_file, encoding='utf-8') as f:
        chunks_data = json.load(f)

    # Convert to DocumentChunk objects
    chunks = []
    for chunk_data in chunks_data:
        metadata = ChunkMetadata(**chunk_data["metadata"])
        chunk = DocumentChunk(text=chunk_data["text"], metadata=metadata)
        chunks.append(chunk)

    logger.info(f"Loaded {len(chunks)} chunks from {chunks_file}")

    # Create embeddings
    pipeline = EmbeddingPipeline(config)
    pipeline.create_embeddings_from_chunks(chunks)

    # Save index
    pipeline.save_index(output_path)

    # Print stats
    stats = pipeline.get_index_stats()
    logger.info(f"Created index with {stats['total_embeddings']} embeddings")
    logger.info(f"Index dimension: {stats['dimension']}")
    logger.info(f"Model used: {stats['model_name']}")


def load_embedding_pipeline(config: dict[str, Any], index_path: Path) -> EmbeddingPipeline:
    """
    Load an embedding pipeline with existing index.
    
    Args:
        config: Configuration dictionary
        index_path: Path to index directory
        
    Returns:
        Loaded EmbeddingPipeline
    """
    pipeline = EmbeddingPipeline(config)
    pipeline.load_index(index_path)
    return pipeline
