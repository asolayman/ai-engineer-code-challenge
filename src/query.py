"""
Query Engine

Handles query processing, similarity search, and chunk retrieval.
Loads FAISS index and embedding model for efficient query processing.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.embed import load_embedding_pipeline
from src.ingest import DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Result of a query with relevant chunks and metadata."""
    query: str
    chunks: list[DocumentChunk]
    similarities: list[float]
    total_chunks_searched: int
    search_time_ms: float


class QueryEngine:
    """Main class for query processing and similarity search."""

    def __init__(self, config: dict[str, Any], index_path: Path | None = None):
        """
        Initialize query engine.
        
        Args:
            config: Configuration dictionary
            index_path: Path to FAISS index (if None, will use config default)
        """
        self.config = config

        # Determine index path
        if index_path is None:
            index_path = Path(config.get("storage", {}).get("index_dir", "./index"))

        self.index_path = index_path

        # Load embedding pipeline with existing index
        try:
            self.embedding_pipeline = load_embedding_pipeline(config, index_path)
            logger.info(f"Loaded query engine with index from {index_path}")
        except Exception as e:
            logger.error(f"Failed to load embedding pipeline: {e}")
            raise

    def search(self, query: str, top_k: int | None = None,
               similarity_threshold: float | None = None) -> QueryResult:
        """
        Search for chunks similar to the query.
        
        Args:
            query: User query text
            top_k: Number of results to return (uses config default if None)
            similarity_threshold: Minimum similarity score (uses config default if None)
            
        Returns:
            QueryResult with relevant chunks and metadata
        """
        import time
        start_time = time.time()

        if not query.strip():
            logger.warning("Empty query provided")
            return QueryResult(
                query=query,
                chunks=[],
                similarities=[],
                total_chunks_searched=0,
                search_time_ms=0.0
            )

        logger.info(f"Processing query: {query[:100]}...")

        # Get search parameters
        if top_k is None:
            top_k = self.embedding_pipeline.embedding_model.config.top_k

        if similarity_threshold is None:
            similarity_threshold = self.embedding_pipeline.embedding_model.config.similarity_threshold

        logger.debug(f"Search parameters: top_k={top_k}, similarity_threshold={similarity_threshold}")

        # Search for similar chunks
        try:
            results = self.embedding_pipeline.search_similar_chunks(query, top_k)

            # Debug logging to see all results before filtering
            logger.debug(f"Raw search results: {len(results)} chunks found")
            for i, (chunk, similarity) in enumerate(results):
                logger.debug(f"  Chunk {i+1}: similarity={similarity:.4f}, file={chunk.metadata.file_name}")

            # Extract chunks and similarities
            chunks = []
            similarities = []

            for chunk, similarity in results:
                logger.debug(f"Comparing similarity {similarity:.4f} >= {similarity_threshold:.4f} = {similarity >= similarity_threshold}")
                if similarity >= similarity_threshold:
                    chunks.append(chunk)
                    similarities.append(similarity)

            # Calculate search time
            search_time_ms = (time.time() - start_time) * 1000

            # Get total chunks in index
            total_chunks = self.embedding_pipeline.faiss_index.get_total_embeddings()

            logger.info(f"Found {len(chunks)} relevant chunks from {total_chunks} total chunks")
            logger.info(f"Search completed in {search_time_ms:.2f}ms")

            return QueryResult(
                query=query,
                chunks=chunks,
                similarities=similarities,
                total_chunks_searched=total_chunks,
                search_time_ms=search_time_ms
            )

        except Exception as e:
            logger.error(f"Error during search: {e}")
            raise

    def get_index_stats(self) -> dict[str, Any]:
        """
        Get statistics about the loaded index.
        
        Returns:
            Dictionary with index statistics
        """
        return self.embedding_pipeline.get_index_stats()

    def validate_index(self) -> bool:
        """
        Validate that the index is properly loaded and functional.
        
        Returns:
            True if index is valid, False otherwise
        """
        try:
            stats = self.get_index_stats()
            if "error" in stats:
                return False

            # Check if index has embeddings
            if stats.get("total_embeddings", 0) == 0:
                logger.warning("Index contains no embeddings")
                return False

            # Test a simple search
            test_query = "test"
            result = self.search(test_query, top_k=1)

            return True

        except Exception as e:
            logger.error(f"Index validation failed: {e}")
            return False


class QueryProcessor:
    """High-level query processor with additional functionality."""

    def __init__(self, config: dict[str, Any], index_path: Path | None = None):
        """
        Initialize query processor.
        
        Args:
            config: Configuration dictionary
            index_path: Path to FAISS index
        """
        self.config = config

        # Determine index path
        if index_path is None:
            index_path = Path(config.get("storage", {}).get("index_dir", "./index"))

        self.index_path = index_path
        self.query_engine = QueryEngine(config, index_path)

        # Load chunk texts if available
        self.chunk_texts = {}
        self._load_chunk_texts()

        logger.info("Initialized QueryProcessor")

    def _load_chunk_texts(self) -> None:
        """Load chunk texts from chunks.json if available."""
        chunks_file = self.index_path / "chunks.json"

        if chunks_file.exists():
            try:
                with open(chunks_file, encoding='utf-8') as f:
                    chunks_data = json.load(f)

                # Create mapping from metadata to text
                for chunk_data in chunks_data:
                    metadata = chunk_data["metadata"]
                    chunk_key = f"{metadata['file_name']}_{metadata['page_number']}_{metadata['chunk_index']}"
                    self.chunk_texts[chunk_key] = chunk_data["text"]

                logger.info(f"Loaded {len(self.chunk_texts)} chunk texts")

            except Exception as e:
                logger.warning(f"Failed to load chunk texts: {e}")

    def process_query(self, query: str, top_k: int | None = None,
                     similarity_threshold: float | None = None) -> QueryResult:
        """
        Process a user query and return relevant chunks.
        
        Args:
            query: User query text
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            QueryResult with relevant chunks and metadata
        """
        # Perform search
        result = self.query_engine.search(query, top_k, similarity_threshold)

        # Enhance chunks with actual text if available
        enhanced_chunks = []
        for chunk in result.chunks:
            # Try to get actual text from chunks.json
            chunk_key = f"{chunk.metadata.file_name}_{chunk.metadata.page_number}_{chunk.metadata.chunk_index}"

            if chunk_key in self.chunk_texts:
                # Create new chunk with actual text
                enhanced_chunk = DocumentChunk(
                    text=self.chunk_texts[chunk_key],
                    metadata=chunk.metadata
                )
                enhanced_chunks.append(enhanced_chunk)
            else:
                # Use original chunk (may have placeholder text)
                enhanced_chunks.append(chunk)

        # Create enhanced result
        enhanced_result = QueryResult(
            query=result.query,
            chunks=enhanced_chunks,
            similarities=result.similarities,
            total_chunks_searched=result.total_chunks_searched,
            search_time_ms=result.search_time_ms
        )

        return enhanced_result

    def format_results(self, result: QueryResult, include_metadata: bool = True) -> str:
        """
        Format query results as a readable string.
        
        Args:
            result: QueryResult to format
            include_metadata: Whether to include chunk metadata
            
        Returns:
            Formatted string representation of results
        """
        if not result.chunks:
            return f"No relevant chunks found for query: '{result.query}'"

        lines = []
        lines.append(f"Query: '{result.query}'")
        lines.append(f"Found {len(result.chunks)} relevant chunks (searched {result.total_chunks_searched} total)")
        lines.append(f"Search time: {result.search_time_ms:.2f}ms")
        lines.append("-" * 50)

        for i, (chunk, similarity) in enumerate(zip(result.chunks, result.similarities, strict=False)):
            lines.append(f"\nChunk {i+1} (similarity: {similarity:.3f}):")

            if include_metadata:
                lines.append(f"  File: {chunk.metadata.file_name}")
                lines.append(f"  Page: {chunk.metadata.page_number}")
                lines.append(f"  Chunk: {chunk.metadata.chunk_index}")
                lines.append(f"  Size: {chunk.metadata.chunk_size} chars")

            # Truncate text if too long
            text = chunk.text
            if len(text) > 500:
                text = text[:500] + "..."

            lines.append(f"  Text: {text}")

        return "\n".join(lines)

    def get_relevant_context(self, result: QueryResult, max_chars: int = 2000) -> str:
        """
        Get relevant context from search results for LLM input.
        
        Args:
            result: QueryResult from search
            max_chars: Maximum characters to include
            
        Returns:
            Formatted context string for LLM
        """
        if not result.chunks:
            return "No relevant information found."

        context_parts = []
        current_length = 0

        for i, chunk in enumerate(result.chunks):
            # Add chunk with metadata
            chunk_text = f"[Document: {chunk.metadata.file_name}, Page: {chunk.metadata.page_number}]\n{chunk.text}\n\n"

            if current_length + len(chunk_text) > max_chars:
                break

            context_parts.append(chunk_text)
            current_length += len(chunk_text)

        if not context_parts:
            return "No relevant information found."

        return "".join(context_parts).strip()


def process_query(query: str, config: dict[str, Any], args: Any) -> QueryResult:
    """
    Main function for query processing.
    
    Args:
        query: User query text
        config: Configuration dictionary
        args: Command line arguments
        
    Returns:
        QueryResult with relevant chunks
    """
    # Override config with command line arguments if provided
    if hasattr(args, 'top_k'):
        config.setdefault('embedding', {})['top_k'] = args.top_k
    if hasattr(args, 'similarity_threshold'):
        config.setdefault('embedding', {})['similarity_threshold'] = args.similarity_threshold

    # Initialize query processor
    index_path = Path(config.get("storage", {}).get("index_dir", "./index"))
    processor = QueryProcessor(config, index_path)

    # Validate index
    if not processor.query_engine.validate_index():
        raise ValueError("Index validation failed. Please ensure index is properly created.")

    # Get search parameters from config
    top_k = config.get("embedding", {}).get("top_k", 5)
    similarity_threshold = config.get("embedding", {}).get("similarity_threshold", 0.7)

    # Process query with parameters
    result = processor.process_query(query, top_k, similarity_threshold)

    # Log results
    logger.info(f"Query processed successfully: {len(result.chunks)} chunks found")

    return result


def format_query_output(result: QueryResult, verbose: bool = False) -> str:
    """
    Format query results for output.
    
    Args:
        result: QueryResult to format
        verbose: Whether to include detailed output
        
    Returns:
        Formatted output string
    """
    if verbose:
        # Detailed formatting without creating QueryProcessor
        if not result.chunks:
            return f"No relevant chunks found for query: '{result.query}'"

        lines = []
        lines.append(f"Query: '{result.query}'")
        lines.append(f"Found {len(result.chunks)} relevant chunks (searched {result.total_chunks_searched} total)")
        lines.append(f"Search time: {result.search_time_ms:.2f}ms")
        lines.append("-" * 50)

        for i, (chunk, similarity) in enumerate(zip(result.chunks, result.similarities, strict=False)):
            lines.append(f"\nChunk {i+1} (similarity: {similarity:.3f}):")
            lines.append(f"  File: {chunk.metadata.file_name}")
            lines.append(f"  Page: {chunk.metadata.page_number}")
            lines.append(f"  Chunk: {chunk.metadata.chunk_index}")
            lines.append(f"  Size: {chunk.metadata.chunk_size} chars")

            # Truncate text if too long
            text = chunk.text
            if len(text) > 500:
                text = text[:500] + "..."

            lines.append(f"  Text: {text}")

        return "\n".join(lines)
    else:
        # Simple output
        if not result.chunks:
            return f"No relevant information found for query: '{result.query}'"

        lines = []
        lines.append(f"Found {len(result.chunks)} relevant chunks:")

        for i, (chunk, similarity) in enumerate(zip(result.chunks, result.similarities, strict=False)):
            lines.append(f"\n{i+1}. [{chunk.metadata.file_name}, p.{chunk.metadata.page_number}] (similarity: {similarity:.3f})")

            # Truncate text
            text = chunk.text
            if len(text) > 200:
                text = text[:200] + "..."
            lines.append(f"   {text}")

        return "\n".join(lines)
