"""
Tests for LLM interface
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ingest import ChunkMetadata, DocumentChunk
from src.llm import (
    BaseLLM,
    LlamaCppLLM,
    LLMConfig,
    LLMInterface,
    LLMResponse,
    OpenAILLM,
    TransformersLLM,
    create_llm_interface,
    format_llm_response,
    generate_answer_from_query,
)
from src.query import QueryResult


class TestLLMConfig:
    """Test LLM configuration"""

    def test_init(self):
        """Test configuration initialization"""
        config = LLMConfig(
            backend="transformers",
            model_path="test-model",
            temperature=0.2,
            max_tokens=1024,
            top_p=0.9,
            repeat_penalty=1.1,
            context_window=4096,
        )

        assert config.backend == "transformers"
        assert config.model_path == "test-model"
        assert config.temperature == 0.2
        assert config.max_tokens == 1024
        assert config.top_p == 0.9
        assert config.repeat_penalty == 1.1
        assert config.context_window == 4096


class TestLLMResponse:
    """Test LLM response dataclass"""

    def test_init(self):
        """Test response initialization"""
        response = LLMResponse(
            answer="This is a test answer.",
            prompt_tokens=100,
            response_tokens=50,
            generation_time_ms=250.0,
            model_used="transformers:test-model",
        )

        assert response.answer == "This is a test answer."
        assert response.prompt_tokens == 100
        assert response.response_tokens == 50
        assert response.generation_time_ms == 250.0
        assert response.model_used == "transformers:test-model"


class TestBaseLLM:
    """Test base LLM class"""

    def test_init(self):
        """Test base LLM initialization"""
        config = LLMConfig(
            backend="test",
            model_path="test-model",
            temperature=0.2,
            max_tokens=1024,
            top_p=0.9,
            repeat_penalty=1.1,
            context_window=4096,
        )

        # BaseLLM should raise NotImplementedError for _load_model
        with pytest.raises(NotImplementedError):
            llm = BaseLLM(config)

    def test_generate_not_implemented(self):
        """Test that generate raises NotImplementedError"""
        config = LLMConfig(
            backend="test",
            model_path="test-model",
            temperature=0.2,
            max_tokens=1024,
            top_p=0.9,
            repeat_penalty=1.1,
            context_window=4096,
        )

        class TestLLM(BaseLLM):
            def _load_model(self):
                pass

        llm = TestLLM(config)

        with pytest.raises(NotImplementedError):
            llm.generate("test prompt")


class TestTransformersLLM:
    """Test transformers LLM implementation"""

    @patch("src.llm.AutoTokenizer")
    @patch("src.llm.AutoModelForCausalLM")
    def test_init(self, mock_model, mock_tokenizer):
        """Test transformers LLM initialization"""
        config = LLMConfig(
            backend="transformers",
            model_path="microsoft/DialoGPT-medium",
            temperature=0.2,
            max_tokens=1024,
            top_p=0.9,
            repeat_penalty=1.1,
            context_window=4096,
        )

        # Mock tokenizer and model
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = MagicMock()
        mock_model_instance.device = "cpu"
        mock_model.from_pretrained.return_value = mock_model_instance

        llm = TransformersLLM(config)

        assert llm.config == config
        assert llm.tokenizer == mock_tokenizer_instance
        assert llm.model == mock_model_instance
        mock_tokenizer.from_pretrained.assert_called_once()
        mock_model.from_pretrained.assert_called_once()

    @patch("src.llm.AutoTokenizer")
    @patch("src.llm.AutoModelForCausalLM")
    def test_generate(self, mock_model, mock_tokenizer):
        """Test transformers generation"""
        config = LLMConfig(
            backend="transformers",
            model_path="microsoft/DialoGPT-medium",
            temperature=0.2,
            max_tokens=1024,
            top_p=0.9,
            repeat_penalty=1.1,
            context_window=4096,
        )

        # Mock tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<|endoftext|>"
        mock_tokenizer_instance.eos_token_id = 50256
        mock_tokenizer_instance.decode.return_value = "Generated response"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Mock model
        mock_model_instance = MagicMock()
        mock_model_instance.device = "cpu"
        mock_model_instance.generate.return_value = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance

        # Mock tokenizer output
        mock_tokenizer_instance.return_value = {
            "input_ids": MagicMock(shape=[1, 10]),
            "attention_mask": MagicMock(),
        }

        llm = TransformersLLM(config)

        # Mock torch operations
        with patch("src.llm.torch") as mock_torch:
            mock_torch.float16 = "float16"
            mock_torch.float32 = "float32"
            mock_torch.cuda.is_available.return_value = False

            response = llm.generate("test prompt")

            assert isinstance(response, LLMResponse)
            assert response.answer == "Generated response"
            assert response.model_used.startswith("transformers:")


class TestLlamaCppLLM:
    """Test llama-cpp LLM implementation"""

    @patch("src.llm.LlamaCppLLM._load_model")
    def test_init(self, mock_load_model):
        """Test llama-cpp LLM initialization"""
        config = LLMConfig(
            backend="llama-cpp",
            model_path="./models/test.gguf",
            temperature=0.2,
            max_tokens=1024,
            top_p=0.9,
            repeat_penalty=1.1,
            context_window=4096,
        )

        # Mock the _load_model method
        mock_load_model.return_value = None

        llm = LlamaCppLLM(config)

        assert llm.config == config
        mock_load_model.assert_called_once()

    @patch("src.llm.LlamaCppLLM._load_model")
    def test_generate(self, mock_load_model):
        """Test llama-cpp generation"""
        config = LLMConfig(
            backend="llama-cpp",
            model_path="./models/test.gguf",
            temperature=0.2,
            max_tokens=1024,
            top_p=0.9,
            repeat_penalty=1.1,
            context_window=4096,
        )

        # Mock the _load_model method
        mock_load_model.return_value = None

        # Mock the model and its response
        mock_model = MagicMock()
        mock_response = {
            "choices": [{"text": "Generated response"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        mock_model.return_value = mock_response
        llm = LlamaCppLLM(config)
        llm.model = mock_model

        response = llm.generate("test prompt")

        assert isinstance(response, LLMResponse)
        assert response.answer == "Generated response"
        assert response.prompt_tokens == 10
        assert response.response_tokens == 5
        assert response.model_used.startswith("llama-cpp:")

    def test_init_import_error(self):
        """Test llama-cpp initialization with import error"""
        config = LLMConfig(
            backend="llama-cpp",
            model_path="./models/test.gguf",
            temperature=0.2,
            max_tokens=1024,
            top_p=0.9,
            repeat_penalty=1.1,
            context_window=4096,
        )

        with patch(
            "src.llm.LlamaCppLLM._load_model",
            side_effect=ImportError("llama-cpp-python not installed"),
        ):
            with pytest.raises(ImportError, match="llama-cpp-python not installed"):
                LlamaCppLLM(config)


class TestOpenAILLM:
    """Test OpenAI LLM implementation"""

    @patch("src.llm.OpenAILLM._load_model")
    def test_init(self, mock_load_model):
        """Test OpenAI LLM initialization"""
        config = LLMConfig(
            backend="openai",
            model_path="sk-test-key",  # API key
            temperature=0.2,
            max_tokens=1024,
            top_p=0.9,
            repeat_penalty=1.1,
            context_window=4096,
        )

        # Mock the _load_model method
        mock_load_model.return_value = None

        llm = OpenAILLM(config)

        assert llm.config == config
        mock_load_model.assert_called_once()

    @patch("src.llm.OpenAILLM._load_model")
    def test_generate(self, mock_load_model):
        """Test OpenAI generation"""
        config = LLMConfig(
            backend="openai",
            model_path="sk-test-key",
            temperature=0.2,
            max_tokens=1024,
            top_p=0.9,
            repeat_penalty=1.1,
            context_window=4096,
        )

        # Mock the _load_model method
        mock_load_model.return_value = None

        # Mock the client and its response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Generated response"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_client.chat.completions.create.return_value = mock_response

        llm = OpenAILLM(config)
        llm.client = mock_client

        response = llm.generate("test prompt")

        assert isinstance(response, LLMResponse)
        assert response.answer == "Generated response"
        assert response.prompt_tokens == 10
        assert response.response_tokens == 5
        assert response.model_used == "openai:gpt-3.5-turbo"

    def test_init_import_error(self):
        """Test OpenAI initialization with import error"""
        config = LLMConfig(
            backend="openai",
            model_path="sk-test-key",
            temperature=0.2,
            max_tokens=1024,
            top_p=0.9,
            repeat_penalty=1.1,
            context_window=4096,
        )

        with patch(
            "src.llm.OpenAILLM._load_model",
            side_effect=ImportError("openai not installed"),
        ):
            with pytest.raises(ImportError, match="openai not installed"):
                OpenAILLM(config)


class TestLLMInterface:
    """Test LLM interface functionality"""

    @patch("src.llm.TransformersLLM")
    def test_init_transformers(self, mock_transformers):
        """Test LLM interface initialization with transformers"""
        config = {
            "llm": {
                "backend": "transformers",
                "model_path": "microsoft/DialoGPT-medium",
                "temperature": 0.2,
                "max_tokens": 1024,
            },
            "prompts": {
                "query_template": "Context: {context}\nQuestion: {question}\nAnswer:"
            },
        }

        mock_llm = MagicMock()
        mock_transformers.return_value = mock_llm

        interface = LLMInterface(config)

        assert interface.config == config
        assert interface.llm == mock_llm
        assert interface.prompts == config["prompts"]
        mock_transformers.assert_called_once()

    @patch("src.llm.LlamaCppLLM")
    def test_init_llama_cpp(self, mock_llama):
        """Test LLM interface initialization with llama-cpp"""
        config = {
            "llm": {
                "backend": "llama-cpp",
                "model_path": "./models/test.gguf",
                "temperature": 0.2,
                "max_tokens": 1024,
            }
        }

        mock_llm = MagicMock()
        mock_llama.return_value = mock_llm

        interface = LLMInterface(config)

        assert interface.llm == mock_llm
        mock_llama.assert_called_once()

    @patch("src.llm.OpenAILLM")
    def test_init_openai(self, mock_openai):
        """Test LLM interface initialization with OpenAI"""
        config = {
            "llm": {
                "backend": "openai",
                "model_path": "sk-test-key",
                "temperature": 0.2,
                "max_tokens": 1024,
            }
        }

        mock_llm = MagicMock()
        mock_openai.return_value = mock_llm

        interface = LLMInterface(config)

        assert interface.llm == mock_llm
        mock_openai.assert_called_once()

    def test_init_invalid_backend(self):
        """Test LLM interface initialization with invalid backend"""
        config = {
            "llm": {
                "backend": "invalid-backend",
                "model_path": "test-model",
                "temperature": 0.2,
                "max_tokens": 1024,
            }
        }

        with pytest.raises(ValueError, match="Unsupported LLM backend"):
            LLMInterface(config)

    @patch("src.llm.TransformersLLM")
    def test_format_prompt(self, mock_transformers):
        """Test prompt formatting"""
        config = {
            "llm": {
                "backend": "transformers",
                "model_path": "test-model",
                "temperature": 0.2,
                "max_tokens": 1024,
            },
            "prompts": {
                "query_template": "Context: {context}\nQuestion: {question}\nAnswer:"
            },
        }

        mock_llm = MagicMock()
        mock_transformers.return_value = mock_llm

        interface = LLMInterface(config)

        prompt = interface.format_prompt("What is the answer?", "This is the context.")

        assert "Context: This is the context." in prompt
        assert "Question: What is the answer?" in prompt
        assert "Answer:" in prompt

    @patch("src.llm.TransformersLLM")
    def test_generate_answer_with_chunks(self, mock_transformers):
        """Test answer generation with chunks"""
        config = {
            "llm": {
                "backend": "transformers",
                "model_path": "test-model",
                "temperature": 0.2,
                "max_tokens": 1024,
            },
            "prompts": {
                "query_template": "Context: {context}\nQuestion: {question}\nAnswer:"
            },
        }

        # Mock LLM
        mock_llm = MagicMock()
        mock_response = LLMResponse(
            answer="This is the answer.",
            prompt_tokens=100,
            response_tokens=50,
            generation_time_ms=250.0,
            model_used="transformers:test-model",
        )
        mock_llm.generate.return_value = mock_response
        mock_transformers.return_value = mock_llm

        interface = LLMInterface(config)

        # Create query result with chunks
        chunks = [
            DocumentChunk(
                text="This is test content.",
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

        query_result = QueryResult(
            query="What is the answer?",
            chunks=chunks,
            similarities=[0.85],
            total_chunks_searched=100,
            search_time_ms=50.0,
        )

        response = interface.generate_answer("What is the answer?", query_result)

        assert isinstance(response, LLMResponse)
        assert response.answer == "This is the answer."
        mock_llm.generate.assert_called_once()

    @patch("src.llm.TransformersLLM")
    def test_generate_answer_no_chunks(self, mock_transformers):
        """Test answer generation with no chunks"""
        config = {
            "llm": {
                "backend": "transformers",
                "model_path": "test-model",
                "temperature": 0.2,
                "max_tokens": 1024,
            },
            "prompts": {"no_answer_template": "No information found."},
        }

        mock_llm = MagicMock()
        mock_transformers.return_value = mock_llm

        interface = LLMInterface(config)

        # Create query result with no chunks
        query_result = QueryResult(
            query="What is the answer?",
            chunks=[],
            similarities=[],
            total_chunks_searched=100,
            search_time_ms=50.0,
        )

        response = interface.generate_answer("What is the answer?", query_result)

        assert isinstance(response, LLMResponse)
        assert response.answer == "No information found."
        assert response.prompt_tokens == 0
        assert response.response_tokens == 0
        assert response.generation_time_ms == 0.0
        mock_llm.generate.assert_not_called()

    @patch("src.llm.TransformersLLM")
    def test_get_model_info(self, mock_transformers):
        """Test getting model information"""
        config = {
            "llm": {
                "backend": "transformers",
                "model_path": "test-model",
                "temperature": 0.2,
                "max_tokens": 1024,
                "context_window": 4096,
            }
        }

        mock_llm = MagicMock()
        mock_llm.config.backend = "transformers"
        mock_llm.config.model_path = "test-model"
        mock_llm.config.temperature = 0.2
        mock_llm.config.max_tokens = 1024
        mock_llm.config.context_window = 4096
        mock_transformers.return_value = mock_llm

        interface = LLMInterface(config)

        info = interface.get_model_info()

        assert info["backend"] == "transformers"
        assert info["model_path"] == "test-model"
        assert info["temperature"] == 0.2
        assert info["max_tokens"] == 1024
        assert info["context_window"] == 4096


class TestLLMFunctions:
    """Test LLM utility functions"""

    @patch("src.llm.LLMInterface")
    def test_create_llm_interface(self, mock_interface_class):
        """Test creating LLM interface"""
        config = {"llm": {"backend": "transformers"}}
        mock_interface = MagicMock()
        mock_interface_class.return_value = mock_interface

        result = create_llm_interface(config)

        assert result == mock_interface
        mock_interface_class.assert_called_once_with(config)

    @patch("src.llm.create_llm_interface")
    def test_generate_answer_from_query(self, mock_create_interface):
        """Test generating answer from query"""
        config = {"llm": {"backend": "transformers"}}

        # Mock LLM interface
        mock_interface = MagicMock()
        mock_response = LLMResponse(
            answer="This is the answer.",
            prompt_tokens=100,
            response_tokens=50,
            generation_time_ms=250.0,
            model_used="transformers:test-model",
        )
        mock_interface.generate_answer.return_value = mock_response
        mock_create_interface.return_value = mock_interface

        # Create query result
        chunks = [
            DocumentChunk(
                text="Test content.",
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

        query_result = QueryResult(
            query="What is the answer?",
            chunks=chunks,
            similarities=[0.85],
            total_chunks_searched=100,
            search_time_ms=50.0,
        )

        answer = generate_answer_from_query("What is the answer?", query_result, config)

        assert answer == "This is the answer."
        mock_interface.generate_answer.assert_called_once_with(
            "What is the answer?", query_result
        )

    def test_format_llm_response_verbose(self):
        """Test verbose LLM response formatting"""
        response = LLMResponse(
            answer="This is the answer.",
            prompt_tokens=100,
            response_tokens=50,
            generation_time_ms=250.0,
            model_used="transformers:test-model",
        )

        formatted = format_llm_response(response, verbose=True)

        assert "=== LLM Response ===" in formatted
        assert "Answer: This is the answer." in formatted
        assert "Model: transformers:test-model" in formatted
        assert "Generation time: 250.00ms" in formatted
        assert "Prompt tokens: 100" in formatted
        assert "Response tokens: 50" in formatted

    def test_format_llm_response_simple(self):
        """Test simple LLM response formatting"""
        response = LLMResponse(
            answer="This is the answer.",
            prompt_tokens=100,
            response_tokens=50,
            generation_time_ms=250.0,
            model_used="transformers:test-model",
        )

        formatted = format_llm_response(response, verbose=False)

        assert formatted == "This is the answer."


if __name__ == "__main__":
    pytest.main([__file__])
