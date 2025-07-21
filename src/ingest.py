"""
Document Ingestion Pipeline

Handles PDF text extraction, cleaning, chunking, and metadata storage.
Supports multiple PDF engines and configurable chunking parameters.
"""

import json
import re
import time
from dataclasses import asdict, dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import pdfplumber
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams

# Import utility functions
from .utils import batch_process, get_logger, log_memory_usage, log_performance

logger = get_logger(__name__)


@dataclass
class ChunkMetadata:
    """Metadata for a text chunk."""
    file_name: str
    page_number: int
    chunk_index: int
    chunk_start: int
    chunk_end: int
    chunk_size: int
    text_length: int


@dataclass
class DocumentChunk:
    """A chunk of text from a document with metadata."""
    text: str
    metadata: ChunkMetadata


class PDFProcessor:
    """Handles PDF text extraction using different engines."""

    def __init__(self, engine: str = "pymupdf"):
        """
        Initialize PDF processor.
        
        Args:
            engine: PDF processing engine ("pymupdf", "pdfminer", "pdfplumber")
        """
        self.engine = engine
        logger.info(f"Initialized PDF processor with engine: {engine}")

    def extract_text(self, pdf_path: Path) -> list[tuple[str, int]]:
        """
        Extract text from PDF with page numbers.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of (text, page_number) tuples
            
        Raises:
            ValueError: If PDF engine is not supported
            FileNotFoundError: If PDF file doesn't exist
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        try:
            if self.engine == "pymupdf":
                return self._extract_with_pymupdf(pdf_path)
            elif self.engine == "pdfminer":
                return self._extract_with_pdfminer(pdf_path)
            elif self.engine == "pdfplumber":
                return self._extract_with_pdfplumber(pdf_path)
            else:
                raise ValueError(f"Unsupported PDF engine: {self.engine}")
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            raise

    def _extract_with_pymupdf(self, pdf_path: Path) -> list[tuple[str, int]]:
        """Extract text using PyMuPDF."""
        pages = []
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                pages.append((text, page_num + 1))
            doc.close()
            return pages
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed for {pdf_path}: {e}")
            raise

    def _extract_with_pdfminer(self, pdf_path: Path) -> list[tuple[str, int]]:
        """Extract text using pdfminer.six."""
        try:
            # Extract all text at once
            output = StringIO()
            extract_text_to_fp(open(pdf_path, 'rb'), output, laparams=LAParams())
            full_text = output.getvalue()
            output.close()

            # For pdfminer, we'll treat the entire document as one page
            # since page-by-page extraction is more complex
            return [(full_text, 1)]
        except Exception as e:
            logger.error(f"PDFMiner extraction failed for {pdf_path}: {e}")
            raise

    def _extract_with_pdfplumber(self, pdf_path: Path) -> list[tuple[str, int]]:
        """Extract text using pdfplumber."""
        pages = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    pages.append((text, page_num + 1))
            return pages
        except Exception as e:
            logger.error(f"PDFPlumber extraction failed for {pdf_path}: {e}")
            raise


class TextCleaner:
    """Handles text cleaning and normalization."""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize text cleaner.
        
        Args:
            config: Configuration dictionary with cleaning parameters
        """
        self.remove_headers = config.get("remove_headers", True)
        self.remove_footers = config.get("remove_footers", True)
        self.normalize_whitespace = config.get("normalize_whitespace", True)
        self.remove_special_chars = config.get("remove_special_chars", False)

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Remove excessive whitespace
        if self.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()

        # Remove headers/footers (simple heuristic)
        if self.remove_headers or self.remove_footers:
            lines = text.split('\n')
            cleaned_lines = []

            for line in lines:
                line = line.strip()
                # Skip lines that look like headers/footers
                if self._is_header_footer(line):
                    continue
                cleaned_lines.append(line)

            text = '\n'.join(cleaned_lines)

        # Remove special characters if requested
        if self.remove_special_chars:
            text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)

        return text

    def _is_header_footer(self, line: str) -> bool:
        """
        Check if a line looks like a header or footer.
        
        Args:
            line: Text line to check
            
        Returns:
            True if line appears to be header/footer
        """
        if not line:
            return False

        # Common header/footer patterns
        patterns = [
            r'^\d+$',  # Page numbers
            r'^[A-Z\s]+$',  # All caps text
            r'^[A-Z][a-z]+\s+\d+$',  # "Page 1" format
            r'^\d+/\d+$',  # "1/10" format
            r'^[A-Z][a-z]+\s+\d{4}$',  # "January 2024" format
        ]

        for pattern in patterns:
            if re.match(pattern, line.strip()):
                return True

        return False


class TextChunker:
    """Handles text chunking with sliding window."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize text chunker.
        
        Args:
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if chunk_overlap >= chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")

        logger.info(f"Initialized chunker: size={chunk_size}, overlap={chunk_overlap}")

    def chunk_text(self, text: str, file_name: str, page_number: int) -> list[DocumentChunk]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            file_name: Name of the source file
            page_number: Page number
            
        Returns:
            List of DocumentChunk objects
        """
        if not text.strip():
            return []

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            # Calculate chunk end
            end = start + self.chunk_size

            # If this is not the last chunk, try to break at a sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                search_start = max(start + self.chunk_size - 100, start)
                search_end = min(end + 50, len(text))

                # Find the last sentence ending
                sentence_end = self._find_sentence_boundary(
                    text[search_start:search_end], search_start
                )

                if sentence_end > start + self.chunk_size * 0.8:  # Only use if reasonable
                    end = sentence_end

            # Extract chunk text
            chunk_text = text[start:end].strip()

            if chunk_text:  # Only add non-empty chunks
                metadata = ChunkMetadata(
                    file_name=file_name,
                    page_number=page_number,
                    chunk_index=chunk_index,
                    chunk_start=start,
                    chunk_end=end,
                    chunk_size=len(chunk_text),
                    text_length=len(text)
                )

                chunks.append(DocumentChunk(text=chunk_text, metadata=metadata))
                chunk_index += 1

            # Move to next chunk with overlap
            start = end - self.chunk_overlap
            if start >= len(text):
                break

        return chunks

    def _find_sentence_boundary(self, text: str, offset: int) -> int:
        """
        Find the last sentence boundary in the given text.
        
        Args:
            text: Text to search
            offset: Offset to add to found position
            
        Returns:
            Position of last sentence boundary
        """
        # Look for sentence endings (. ! ?)
        sentence_endings = ['.', '!', '?', '\n\n']

        for i in range(len(text) - 1, -1, -1):
            if text[i] in sentence_endings:
                return offset + i + 1

        return offset + len(text)


class DocumentIngester:
    """Main class for document ingestion pipeline."""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize document ingester.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.pdf_processor = PDFProcessor(config.get("pdf", {}).get("engine", "pymupdf"))
        self.text_cleaner = TextCleaner(config)

        pdf_config = config.get("pdf", {})
        chunk_size = pdf_config.get("chunk_size", 1000)
        chunk_overlap = pdf_config.get("chunk_overlap", 200)

        # Debug logging
        logger.debug(f"PDF config: {pdf_config}")
        logger.debug(f"Chunk size: {chunk_size} (type: {type(chunk_size)})")
        logger.debug(f"Chunk overlap: {chunk_overlap} (type: {type(chunk_overlap)})")

        self.chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        logger.info("Initialized DocumentIngester")

    @log_performance
    def ingest_documents(self, documents_path: Path) -> list[DocumentChunk]:
        """
        Ingest all PDF documents from the given path.
        
        Args:
            documents_path: Path to directory containing PDF files
            
        Returns:
            List of all document chunks
            
        Raises:
            ValueError: If documents_path doesn't exist or contains no PDFs
        """
        if not documents_path.exists():
            raise ValueError(f"Documents path does not exist: {documents_path}")

        if not documents_path.is_dir():
            raise ValueError(f"Documents path is not a directory: {documents_path}")

        # Find all PDF files
        pdf_files = list(documents_path.glob("*.pdf"))
        if not pdf_files:
            raise ValueError(f"No PDF files found in {documents_path}")

        logger.info(f"Found {len(pdf_files)} PDF files to process")
        log_memory_usage(logger, "Before document processing")

        # Get batch size from config
        batch_size = self.config.get("system", {}).get("batch_size", 32)

        # Process documents in batches
        all_chunks = batch_process(
            items=pdf_files,
            batch_size=batch_size,
            process_func=self._process_batch,
            logger=logger,
            description="Document ingestion"
        )

        logger.info(f"Total chunks created: {len(all_chunks)}")
        log_memory_usage(logger, "After document processing")

        return all_chunks

    def _process_batch(self, pdf_files: list[Path]) -> list[DocumentChunk]:
        """
        Process a batch of PDF files.
        
        Args:
            pdf_files: List of PDF file paths
            
        Returns:
            List of chunks from all documents in the batch
        """
        batch_chunks = []

        for pdf_file in pdf_files:
            try:
                chunks = self._process_single_document(pdf_file)
                batch_chunks.extend(chunks)
                logger.debug(f"Processed {pdf_file.name}: {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Failed to process {pdf_file.name}: {e}")
                continue

        return batch_chunks

    def _process_single_document(self, pdf_path: Path) -> list[DocumentChunk]:
        """
        Process a single PDF document.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of chunks from this document
        """
        # Extract text from PDF
        pages = self.pdf_processor.extract_text(pdf_path)

        all_chunks = []

        # Process each page
        for page_text, page_number in pages:
            # Clean the text
            cleaned_text = self.text_cleaner.clean_text(page_text)

            if not cleaned_text.strip():
                continue

            # Chunk the text
            chunks = self.chunker.chunk_text(
                cleaned_text,
                pdf_path.name,
                page_number
            )

            all_chunks.extend(chunks)

        return all_chunks

    @log_performance
    def save_chunks(self, chunks: list[DocumentChunk], output_path: Path) -> None:
        """
        Save chunks and metadata to disk.
        
        Args:
            chunks: List of document chunks
            output_path: Path to save chunks
        """
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving {len(chunks)} chunks to {output_path}")
        log_memory_usage(logger, "Before saving chunks")

        # Convert chunks to serializable format
        chunks_data = []
        for chunk in chunks:
            chunk_dict = {
                "text": chunk.text,
                "metadata": asdict(chunk.metadata)
            }
            chunks_data.append(chunk_dict)

        # Save chunks to JSON file
        chunks_file = output_path / "chunks.json"
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)

        # Create ingestion summary
        summary = {
            "total_chunks": len(chunks),
            "total_files": len(set(chunk.metadata.file_name for chunk in chunks)),
            "total_pages": len(set((chunk.metadata.file_name, chunk.metadata.page_number) for chunk in chunks)),
            "chunk_size": self.chunker.chunk_size,
            "chunk_overlap": self.chunker.chunk_overlap,
            "pdf_engine": self.pdf_processor.engine,
            "ingestion_time": time.time(),
            "file_stats": {}
        }

        # Calculate file statistics
        file_stats = {}
        for chunk in chunks:
            file_name = chunk.metadata.file_name
            if file_name not in file_stats:
                file_stats[file_name] = {
                    "chunks": 0,
                    "pages": set(),
                    "total_text_length": 0
                }
            file_stats[file_name]["chunks"] += 1
            file_stats[file_name]["pages"].add(chunk.metadata.page_number)
            file_stats[file_name]["total_text_length"] += len(chunk.text)

        # Convert sets to lists for JSON serialization
        for file_name, stats in file_stats.items():
            stats["pages"] = list(stats["pages"])
            summary["file_stats"][file_name] = stats

        # Save summary
        summary_file = output_path / "ingestion_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved chunks to {chunks_file}")
        logger.info(f"Saved summary to {summary_file}")
        log_memory_usage(logger, "After saving chunks")


def ingest_documents(documents_path: str, config: dict[str, Any], args: Any) -> None:
    """
    Main function for document ingestion.
    
    Args:
        documents_path: Path to documents directory
        config: Configuration dictionary
        args: Command line arguments
    """
    documents_path = Path(documents_path)

    # Override config with command line arguments if provided
    if hasattr(args, 'chunk_size') and args.chunk_size is not None:
        config.setdefault('pdf', {})['chunk_size'] = args.chunk_size
    if hasattr(args, 'chunk_overlap') and args.chunk_overlap is not None:
        config.setdefault('pdf', {})['chunk_overlap'] = args.chunk_overlap

    # Initialize ingester
    ingester = DocumentIngester(config)

    # Process documents
    chunks = ingester.ingest_documents(documents_path)

    # Save results
    output_path = Path(config.get("storage", {}).get("index_dir", "./index"))
    ingester.save_chunks(chunks, output_path)

    logger.info("Document ingestion completed successfully")
