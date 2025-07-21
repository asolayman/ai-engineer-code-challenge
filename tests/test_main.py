"""
Tests for main.py CLI functionality
"""

import os

# Import the functions we want to test
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from main import load_config, validate_ingest_args, validate_query_args


class TestConfigLoading:
    """Test configuration loading functionality"""

    def test_load_config_success(self):
        """Test successful config loading"""
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
embedding:
  model_name: "test-model"
llm:
  backend: "test-backend"
            """)
            config_path = f.name

        try:
            config = load_config(config_path)
            assert config['embedding']['model_name'] == "test-model"
            assert config['llm']['backend'] == "test-backend"
        finally:
            os.unlink(config_path)

    def test_load_config_file_not_found(self):
        """Test config loading with non-existent file"""
        with pytest.raises(SystemExit):
            load_config("non_existent_config.yaml")

    def test_load_config_invalid_yaml(self):
        """Test config loading with invalid YAML"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_path = f.name

        try:
            with pytest.raises(SystemExit):
                load_config(config_path)
        finally:
            os.unlink(config_path)


class TestIngestValidation:
    """Test ingest argument validation"""

    def test_validate_ingest_args_success(self):
        """Test successful ingest argument validation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test directory with PDF files
            pdf_dir = Path(temp_dir) / "test_pdfs"
            pdf_dir.mkdir()

            # Create a dummy PDF file
            (pdf_dir / "test.pdf").touch()

            args = MagicMock()
            args.documents = str(pdf_dir)

            # Should not raise any exception
            validate_ingest_args(args)

    def test_validate_ingest_args_directory_not_exists(self):
        """Test validation with non-existent directory"""
        args = MagicMock()
        args.documents = "/non/existent/path"

        with pytest.raises(ValueError, match="does not exist"):
            validate_ingest_args(args)

    def test_validate_ingest_args_not_directory(self):
        """Test validation with file instead of directory"""
        with tempfile.NamedTemporaryFile() as temp_file:
            args = MagicMock()
            args.documents = temp_file.name

            with pytest.raises(ValueError, match="is not a directory"):
                validate_ingest_args(args)

    def test_validate_ingest_args_no_pdfs(self):
        """Test validation with directory containing no PDFs"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create empty directory
            args = MagicMock()
            args.documents = temp_dir

            # Should not raise exception, but should log warning
            with patch('main.logger') as mock_logger:
                validate_ingest_args(args)
                mock_logger.warning.assert_called()


class TestQueryValidation:
    """Test query argument validation"""

    def test_validate_query_args_success(self):
        """Test successful query argument validation"""
        # Create index directory
        with tempfile.TemporaryDirectory() as temp_dir:
            index_dir = Path(temp_dir) / "index"
            index_dir.mkdir()

            args = MagicMock()
            args.query = "What is the answer?"

            with patch('main.Path') as mock_path:
                mock_path.return_value.exists.return_value = True
                validate_query_args(args)

    def test_validate_query_args_empty_query(self):
        """Test validation with empty query"""
        args = MagicMock()
        args.query = ""

        with pytest.raises(ValueError, match="cannot be empty"):
            validate_query_args(args)

    def test_validate_query_args_whitespace_query(self):
        """Test validation with whitespace-only query"""
        args = MagicMock()
        args.query = "   "

        with pytest.raises(ValueError, match="cannot be empty"):
            validate_query_args(args)

    def test_validate_query_args_no_index(self):
        """Test validation when index directory doesn't exist"""
        args = MagicMock()
        args.query = "What is the answer?"

        with patch('main.Path') as mock_path:
            mock_path.return_value.exists.return_value = False
            with pytest.raises(ValueError, match="Index directory does not exist"):
                validate_query_args(args)


class TestMainCLI:
    """Test main CLI functionality"""

    @patch('main.load_config')
    @patch('main.validate_ingest_args')
    @patch('src.ingest.ingest_documents')
    @patch('src.embed.create_embeddings_from_chunks_file')
    def test_main_ingest_mode(self, mock_create_embeddings, mock_ingest, mock_validate, mock_load_config):
        """Test main function with ingest mode"""
        mock_load_config.return_value = {}

        with patch('sys.argv', ['main.py', '--mode', 'ingest', '--documents', './test']):
            with patch('main.validate_ingest_args') as mock_validate:
                with patch('main.logger') as mock_logger:
                    from main import main
                    main()
                    mock_validate.assert_called_once()
                    # Check that both logging messages were called
                    mock_logger.info.assert_any_call("Starting document ingestion...")
                    mock_logger.info.assert_any_call("Creating embeddings from ingested chunks...")
                    mock_ingest.assert_called_once()
                    mock_create_embeddings.assert_called_once()

    @patch('main.load_config')
    @patch('main.validate_query_args')
    @patch('src.query.process_query')
    @patch('src.query.format_query_output')
    @patch('src.llm.generate_answer_from_query')
    def test_main_query_mode(self, mock_generate_answer, mock_format_output, mock_process_query, mock_validate, mock_load_config):
        """Test main function with query mode"""
        mock_load_config.return_value = {}

        # Mock QueryResult
        from src.query import QueryResult
        mock_result = QueryResult(
            query="test query",
            chunks=[],
            similarities=[],
            total_chunks_searched=0,
            search_time_ms=0.0
        )
        mock_process_query.return_value = mock_result
        mock_format_output.return_value = "formatted_output"
        mock_generate_answer.return_value = "Generated answer"

        with patch('sys.argv', ['main.py', '--mode', 'query', '--query', 'test query']):
            with patch('main.validate_query_args') as mock_validate:
                with patch('main.logger') as mock_logger:
                    with patch('builtins.print') as mock_print:
                        from main import main
                        main()
                        mock_validate.assert_called_once()
                        mock_logger.info.assert_any_call("Starting query processing...")
                        mock_logger.info.assert_any_call("Generating answer using LLM...")
                        mock_process_query.assert_called_once()
                        mock_generate_answer.assert_called_once()
                        mock_print.assert_called_once_with("Generated answer")


if __name__ == "__main__":
    pytest.main([__file__])
