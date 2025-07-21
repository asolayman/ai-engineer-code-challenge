"""
LLM Interface

Handles local LLM loading, prompt formatting, and answer generation.
Supports multiple backends: transformers, llama-cpp, and OpenAI (optional).
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.query import QueryResult

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM settings."""

    backend: str
    model_path: str
    temperature: float
    max_tokens: int
    top_p: float
    repeat_penalty: float
    context_window: int


@dataclass
class LLMResponse:
    """Response from LLM with metadata."""

    answer: str
    prompt_tokens: int
    response_tokens: int
    generation_time_ms: float
    model_used: str


class BaseLLM:
    """Base class for LLM implementations."""

    def __init__(self, config: LLMConfig):
        """
        Initialize LLM with configuration.

        Args:
            config: LLM configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self._load_model()
        logger.info(f"Initialized {config.backend} LLM: {config.model_path}")

    def _load_model(self) -> None:
        """Load the LLM model. To be implemented by subclasses."""
        raise NotImplementedError

    def generate(self, prompt: str) -> LLMResponse:
        """
        Generate response from prompt.

        Args:
            prompt: Input prompt

        Returns:
            LLMResponse with answer and metadata
        """
        raise NotImplementedError


class TransformersLLM(BaseLLM):
    """LLM implementation using transformers library."""

    def _load_model(self) -> None:
        """Load transformers model and tokenizer."""
        try:
            logger.info(f"Loading transformers model: {self.config.model_path}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path, trust_remote_code=True
            )

            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=torch.float16
                if torch.cuda.is_available()
                else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )

            logger.info(f"Loaded transformers model on {self.model.device}")

        except Exception as e:
            logger.error(f"Failed to load transformers model: {e}")
            raise

    def generate(self, prompt: str) -> LLMResponse:
        """
        Generate response using transformers.

        Args:
            prompt: Input prompt

        Returns:
            LLMResponse with answer and metadata
        """
        start_time = time.time()

        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.context_window,
            )

            # Move to same device as model
            if hasattr(self.model, "device"):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            logger.debug(f"Input tokens: {inputs['input_ids'].shape[1]}")

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    repetition_penalty=self.config.repeat_penalty,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True,
                )

            logger.debug(f"Output tokens: {outputs.shape[1]}")

            # Decode response - get only the new tokens
            input_length = inputs["input_ids"].shape[1]
            response_tokens = outputs[0][input_length:]

            logger.debug(f"Response tokens shape: {response_tokens.shape}")
            logger.debug(f"Response tokens: {response_tokens}")

            # Decode the response tokens
            answer = self.tokenizer.decode(response_tokens, skip_special_tokens=True)

            logger.debug(f"Decoded answer: '{answer}'")

            # Calculate metadata
            generation_time_ms = (time.time() - start_time) * 1000
            prompt_tokens = input_length
            response_tokens_count = len(response_tokens)

            return LLMResponse(
                answer=answer.strip(),
                prompt_tokens=prompt_tokens,
                response_tokens=response_tokens_count,
                generation_time_ms=generation_time_ms,
                model_used=f"transformers:{self.config.model_path}",
            )

        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise


class LlamaCppLLM(BaseLLM):
    """LLM implementation using llama-cpp-python."""

    def _load_model(self) -> None:
        """Load llama-cpp model."""
        try:
            logger.info(f"Loading llama-cpp model: {self.config.model_path}")

            # Import llama-cpp-python
            from llama_cpp import Llama

            self.model = Llama(
                model_path=self.config.model_path,
                n_ctx=self.config.context_window,
                n_threads=4,  # Configurable
                n_gpu_layers=0,  # CPU only for now
            )

            logger.info("Loaded llama-cpp model")

        except ImportError:
            logger.error(
                "llama-cpp-python not installed. Install with: pip install llama-cpp-python"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load llama-cpp model: {e}")
            raise

    def generate(self, prompt: str) -> LLMResponse:
        """
        Generate response using llama-cpp.

        Args:
            prompt: Input prompt

        Returns:
            LLMResponse with answer and metadata
        """
        start_time = time.time()

        try:
            # Generate response
            response = self.model(
                prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                repeat_penalty=self.config.repeat_penalty,
                stop=["</s>", "\n\n\n"],  # Common stop tokens
            )

            answer = response["choices"][0]["text"].strip()

            # Calculate metadata
            generation_time_ms = (time.time() - start_time) * 1000
            prompt_tokens = response["usage"]["prompt_tokens"]
            response_tokens = response["usage"]["completion_tokens"]

            return LLMResponse(
                answer=answer,
                prompt_tokens=prompt_tokens,
                response_tokens=response_tokens,
                generation_time_ms=generation_time_ms,
                model_used=f"llama-cpp:{self.config.model_path}",
            )

        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise


class OpenAILLM(BaseLLM):
    """LLM implementation using OpenAI API (optional)."""

    def _load_model(self) -> None:
        """Initialize OpenAI client."""
        try:
            import openai

            self.client = openai.OpenAI(
                api_key=self.config.model_path  # model_path contains API key
            )
            logger.info("Initialized OpenAI client")

        except ImportError:
            logger.error("openai not installed. Install with: pip install openai")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise

    def generate(self, prompt: str) -> LLMResponse:
        """
        Generate response using OpenAI API.

        Args:
            prompt: Input prompt

        Returns:
            LLMResponse with answer and metadata
        """
        start_time = time.time()

        try:
            # Generate response
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )

            answer = response.choices[0].message.content.strip()

            # Calculate metadata
            generation_time_ms = (time.time() - start_time) * 1000
            prompt_tokens = response.usage.prompt_tokens
            response_tokens = response.usage.completion_tokens

            return LLMResponse(
                answer=answer,
                prompt_tokens=prompt_tokens,
                response_tokens=response_tokens,
                generation_time_ms=generation_time_ms,
                model_used="openai:gpt-3.5-turbo",
            )

        except Exception as e:
            logger.error(f"Error during OpenAI generation: {e}")
            raise


class LLMInterface:
    """Main interface for LLM operations."""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize LLM interface.

        Args:
            config: Configuration dictionary
        """
        self.config = config

        # Create LLM config
        llm_config = LLMConfig(
            backend=config.get("llm", {}).get("backend", "transformers"),
            model_path=config.get("llm", {}).get(
                "model_path", "microsoft/DialoGPT-medium"
            ),
            temperature=config.get("llm", {}).get("temperature", 0.2),
            max_tokens=config.get("llm", {}).get("max_tokens", 1024),
            top_p=config.get("llm", {}).get("top_p", 0.9),
            repeat_penalty=config.get("llm", {}).get("repeat_penalty", 1.1),
            context_window=config.get("llm", {}).get("context_window", 4096),
        )

        # Initialize appropriate LLM
        if llm_config.backend == "transformers":
            self.llm = TransformersLLM(llm_config)
        elif llm_config.backend == "llama-cpp":
            self.llm = LlamaCppLLM(llm_config)
        elif llm_config.backend == "openai":
            self.llm = OpenAILLM(llm_config)
        else:
            raise ValueError(f"Unsupported LLM backend: {llm_config.backend}")

        # Load prompt templates
        self.prompts = config.get("prompts", {})

        logger.info("Initialized LLMInterface")

    def format_prompt(self, query: str, context: str) -> str:
        """
        Format prompt with query and context.

        Args:
            query: User query
            context: Retrieved document context

        Returns:
            Formatted prompt
        """
        # Mistral instruction format
        template = self.prompts.get(
            "query_template",
            "<s>[INST] Based on the following context, answer the question. If the context doesn't contain enough information to answer the question, respond with 'I don't have enough information to answer this question.'\n\nContext:\n{context}\n\nQuestion: {question} [/INST]",
        )

        return template.format(context=context, question=query)

    def generate_answer(self, query: str, query_result: QueryResult) -> LLMResponse:
        """
        Generate answer from query and retrieved chunks.

        Args:
            query: User query
            query_result: QueryResult with retrieved chunks

        Returns:
            LLMResponse with generated answer
        """
        if not query_result.chunks:
            # No relevant chunks found
            no_answer_template = self.prompts.get(
                "no_answer_template",
                "I don't have enough information to answer this question based on the available documents.",
            )

            return LLMResponse(
                answer=no_answer_template,
                prompt_tokens=0,
                response_tokens=0,
                generation_time_ms=0.0,
                model_used=self.llm.config.backend,
            )

        # Format context from chunks
        context_parts = []
        for chunk in query_result.chunks:
            context_parts.append(
                f"[Document: {chunk.metadata.file_name}, Page: {chunk.metadata.page_number}]\n{chunk.text}"
            )

        context = "\n\n".join(context_parts)

        # Format prompt
        prompt = self.format_prompt(query, context)

        # Generate answer
        response = self.llm.generate(prompt)

        logger.info(f"Generated answer in {response.generation_time_ms:.2f}ms")
        logger.info(
            f"Used {response.prompt_tokens} prompt tokens, {response.response_tokens} response tokens"
        )

        return response

    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        return {
            "backend": self.llm.config.backend,
            "model_path": self.llm.config.model_path,
            "temperature": self.llm.config.temperature,
            "max_tokens": self.llm.config.max_tokens,
            "context_window": self.llm.config.context_window,
        }


def create_llm_interface(config: dict[str, Any]) -> LLMInterface:
    """
    Create LLM interface from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        LLMInterface instance
    """
    return LLMInterface(config)


def generate_answer_from_query(
    query: str, query_result: QueryResult, config: dict[str, Any]
) -> str:
    """
    Generate answer from query and query result.

    Args:
        query: User query
        query_result: QueryResult with retrieved chunks
        config: Configuration dictionary

    Returns:
        Generated answer string
    """
    # Create LLM interface
    llm_interface = create_llm_interface(config)

    # Generate answer
    response = llm_interface.generate_answer(query, query_result)

    return response.answer


def format_llm_response(response: LLMResponse, verbose: bool = False) -> str:
    """
    Format LLM response for output.

    Args:
        response: LLMResponse to format
        verbose: Whether to include metadata

    Returns:
        Formatted output string
    """
    if verbose:
        lines = []
        lines.append("=== LLM Response ===")
        lines.append(f"Answer: {response.answer}")
        lines.append(f"Model: {response.model_used}")
        lines.append(f"Generation time: {response.generation_time_ms:.2f}ms")
        lines.append(f"Prompt tokens: {response.prompt_tokens}")
        lines.append(f"Response tokens: {response.response_tokens}")
        return "\n".join(lines)
    else:
        return response.answer
