"""
AI00 Server client for model interaction.
Handles chat completions, embeddings, and model loading.
"""

import httpx
from openai import AsyncOpenAI
from typing import Optional, Dict, Any, List


class AI00Client:
    """Client for interacting with ai00-server."""

    def __init__(self, base_url: str = "http://127.0.0.1:65530"):
        self.base_url = base_url.rstrip("/")
        self.client = AsyncOpenAI(
            base_url=f"{self.base_url}/api/oai", api_key="not-needed"
        )
        self.current_model: Optional[str] = None

    # Why need to load model when ai00-server config already does that?
    # Ans: It's probably for future use case when running in background.
    async def load_model(
        self,
        model_path: str,
        tokenizer_path: str = "assets/tokenizer/rwkv_vocab_v20230424.json",
        quant_type: str = "NF4",
        quant: int = 0,
        max_batch: int = 8,
        token_chunk_size: int = 128,
        precision: str = "Fp16",
        embed_device: str = "Gpu",
        adapter: Optional[Dict] = None,
        timeout: float = 300.0,
    ) -> bool:
        payload = {
            "model_path": model_path,
            "tokenizer_path": tokenizer_path,
            "quant_type": quant_type,
            "quant": quant,
            "max_batch": max_batch,
            "token_chunk_size": token_chunk_size,
            "precision": precision,
            "embed_device": embed_device,
        }

        if adapter:
            payload["adapter"] = adapter

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/admin/models/load", json=payload, timeout=timeout
                )
                response.raise_for_status()
                self.current_model = model_path
                return True
            except Exception as e:
                print(f"Failed to load model: {e}")
                return False

    # Default stop sequences for RWKV7 to prevent rambling
    DEFAULT_STOP_SEQUENCES = ["User:", "Assistant:", "System:"]

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        sampler_config: Optional[Dict[str, Any]] = None,
        max_tokens: int = 1000,
        stream: bool = False,
        stop: Optional[List[str]] = None,
    ) -> str:
        """
        Send a chat completion request to server.

        Args:
            messages: List of message dicts with 'role' and 'content'
            system_prompt: Optional system prompt to prepend
            sampler_config: Sampler configuration dict
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            stop: List of stop sequences (uses defaults for RWKV7 if not provided)

        Returns:
            Generated text response
        """
        # Build messages list
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        # Build request kwargs
        kwargs: Dict[str, Any] = {
            "model": "default",
            "messages": full_messages,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        # Use default stop sequences for RWKV7 if none provided
        if stop is None:
            stop = self.DEFAULT_STOP_SEQUENCES
        kwargs["stop"] = stop

        # Use extra_body to pass custom ai00-server parameters
        extra_body: Dict[str, Any] = {}
        if sampler_config:
            extra_body["sampler_override"] = sampler_config

        if extra_body:
            kwargs["extra_body"] = extra_body

        try:
            if stream:
                response_text = ""
                stream_response = await self.client.chat.completions.create(**kwargs)
                async for chunk in stream_response:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        response_text += content
                        print(content, end="", flush=True)
                print()  # Newline after streaming
                # Validate streaming response
                is_valid, cleaned = self.validate_response(response_text)
                if not is_valid:
                    print(f"\n[Filtered: {cleaned}]")
                return cleaned if is_valid else response_text
            else:
                response = await self.client.chat.completions.create(**kwargs)
                response_text = response.choices[0].message.content
                # Validate non-streaming response
                is_valid, cleaned = self.validate_response(response_text)
                return cleaned
        except Exception as e:
            error_msg = f"Chat completion failed: {e}"
            print(error_msg)
            return error_msg

    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        sampler_config: Optional[Dict[str, Any]] = None,
        max_tokens: int = 1000,
        stream: bool = False,
    ) -> str:
        """
        Simple text generation with a single prompt.

        Args:
            prompt: The input prompt
            system_prompt: Optional system context
            sampler_config: Sampler configuration
            max_tokens: Maximum tokens to generate
            stream: Whether to stream output

        Returns:
            Generated text
        """
        messages = [{"role": "user", "content": prompt}]
        return await self.chat_completion(
            messages=messages,
            system_prompt=system_prompt,
            sampler_config=sampler_config,
            max_tokens=max_tokens,
            stream=stream,
        )

    async def get_embeddings(self, text: str, layer: int = 0) -> Optional[List[float]]:
        """Get embeddings for text."""
        try:
            response = await self.client.embeddings.create(
                model="default", input=text, encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Embedding failed: {e}")
            return None

    # Another one which doesn't use case for now
    async def unload_model(self) -> bool:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/admin/models/unload")
                response.raise_for_status()
                self.current_model = None
                return True
        except Exception as e:
            print(f"Failed to unload model: {e}")
            return False

    def validate_response(self, response: str) -> tuple[bool, str]:
        """
        Validate response to detect rambling or system prompt leakage.

        Returns:
            (is_valid, cleaned_response): Tuple indicating if valid and cleaned text
        """
        if not response:
            return True, response

        # Patterns indicating rambling about system instructions
        rambling_patterns = [
            "You are",
            "system prompt",
            "instruction",
            "As an AI",
            "As a language model",
            "my programming",
            "I am programmed to",
            "I cannot",
            "I must follow",
        ]

        lines = response.split("\n")
        cleaned_lines = []
        rambling_detected = False

        for line in lines:
            # Check if line starts with rambling pattern
            if any(line.strip().startswith(pattern) for pattern in rambling_patterns):
                rambling_detected = True
                continue
            cleaned_lines.append(line)

        cleaned_response = "\n".join(cleaned_lines).strip()

        # If response is too short after cleaning, it was all rambling
        if rambling_detected and len(cleaned_response) < 10:
            return (
                False,
                "[Response was filtered due to system prompt rambling. Please try again.]",
            )

        return True, cleaned_response

    async def check_health(self) -> bool:
        """Check if ai00-server is running and ready to serve."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/api/oai/models", timeout=5.0
                )
                return response.status_code == 200
        except:
            return False
