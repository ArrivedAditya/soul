"""
LangChain-compatible wrapper for ai00-server.

Provides a BaseChatModel implementation that bridges LangChain's
interface with ai00-server's OpenAI-compatible API.
"""

import httpx
from typing import Any, Dict, List, Optional, Iterator
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.callbacks import CallbackManagerForLLMRun
from pydantic import Field


class ChatAI00(BaseChatModel):
    """
    LangChain-compatible chat model for ai00-server (RWKV).

    This wrapper provides:
    - Standard LangChain interface (invoke, stream, bind_tools)
    - Support for ai00-server sampler configurations
    - RWKV-specific stop sequences
    - Response validation (rambling detection)

    Usage:
        from models.langchain_ai00 import ChatAI00
        from langchain.prompts import ChatPromptTemplate

        llm = ChatAI00()
        response = llm.invoke([HumanMessage(content="Hello")])

        # Or with LCEL
        chain = ChatPromptTemplate.from_messages([("system", "You are {name}"), ("user", "Hello {name}")]) | llm
        response = chain.invoke({"name": "Soul"})
    """

    base_url: str = Field(default="http://127.0.0.1:65530")
    model_name: str = Field(default="default")
    max_tokens: int = Field(default=1000)
    temperature: float = Field(default=1.0)
    top_p: Optional[float] = Field(default=None)
    stop: Optional[List[str]] = Field(default=None)
    sampler_config: Optional[Dict[str, Any]] = Field(default=None)

    DEFAULT_STOP_SEQUENCES: List[str] = ["User:", "Assistant:", "System:"]

    @property
    def _llm_type(self) -> str:
        return "ai00-chat"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

    def _convert_messages(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """Convert LangChain BaseMessage objects to ai00-server format."""
        result = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            elif isinstance(msg, SystemMessage):
                role = "system"
            else:
                role = msg.type or "user"
            result.append({"role": role, "content": msg.content})
        return result

    def _get_sampler_params(self) -> Dict[str, Any]:
        """Get sampler parameters for ai00-server."""
        if self.sampler_config:
            return self.sampler_config

        params = {"type": "Nucleus", "temperature": self.temperature}

        if self.top_p is not None:
            params["top_p"] = self.top_p

        if self.stop:
            params["stop"] = self.stop

        return params

    def _create_client(self) -> httpx.AsyncClient:
        """Create async HTTP client for ai00-server."""
        return httpx.AsyncClient(
            base_url=f"{self.base_url}/api/oai",
            headers={"Authorization": "Bearer not-needed"},
            timeout=60.0,
        )

    def _validate_response(self, response: str) -> tuple[bool, str]:
        """Validate response to detect rambling or system prompt leakage."""
        if not response:
            return True, response

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
            if any(line.strip().startswith(pattern) for pattern in rambling_patterns):
                rambling_detected = True
                continue
            cleaned_lines.append(line)

        cleaned_response = "\n".join(cleaned_lines).strip()

        if rambling_detected and len(cleaned_response) < 10:
            return False, "[Response filtered due to system prompt rambling]"

        return True, cleaned_response

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generate method - called by LangChain."""
        converted_messages = self._convert_messages(messages)

        stop_sequences = stop or self.stop or self.DEFAULT_STOP_SEQUENCES

        sampler_params = self._get_sampler_params()
        extra_body = {"sampler_override": sampler_params}

        request_params = {
            "model": self.model_name,
            "messages": converted_messages,
            "max_tokens": self.max_tokens,
            "stop": stop_sequences,
            "extra_body": extra_body,
        }

        async with self._create_client() as client:
            try:
                response = await client.post(
                    "/chat/completions",
                    json=request_params,
                )
                response.raise_for_status()
                result = response.json()

                content = result["choices"][0]["message"]["content"]
                is_valid, cleaned = self._validate_response(content)

                if not is_valid:
                    content = cleaned

                ai_message = AIMessage(content=content)
                generation = ChatGeneration(message=ai_message)

                return ChatResult(generations=[generation])

            except httpx.HTTPStatusError as e:
                raise ValueError(
                    f"ai00-server error: {e.response.status_code} - {e.response.text}"
                )
            except Exception as e:
                raise ValueError(f"Chat completion failed: {str(e)}")

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Synchronous generate - wraps async for compatibility."""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            raise RuntimeError(
                "ChatAI00._generate called from async context. Use ainvoke() instead."
            )
        except RuntimeError:
            pass

        return asyncio.run(self._agenerate(messages, stop, run_manager, **kwargs))

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ):
        """Stream generate - yields ChatResults incrementally.

        Note: This is a sync wrapper. For true async streaming, use astream().
        """
        converted_messages = self._convert_messages(messages)
        stop_sequences = stop or self.stop or self.DEFAULT_STOP_SEQUENCES
        sampler_params = self._get_sampler_params()

        request_params = {
            "model": self.model_name,
            "messages": converted_messages,
            "max_tokens": self.max_tokens,
            "stop": stop_sequences,
            "stream": True,
            "extra_body": {"sampler_override": sampler_params},
        }

        import asyncio
        from typing import AsyncGenerator

        async def stream_generator() -> AsyncGenerator[str, None]:
            async with self._create_client() as client:
                async with client.stream(
                    "POST", "/chat/completions", json=request_params
                ) as response:
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                break
                            import json

                            chunk = json.loads(data)
                            content = (
                                chunk["choices"][0].get("delta", {}).get("content", "")
                            )
                            if content:
                                yield content

        try:
            loop = asyncio.get_running_loop()
            raise RuntimeError(
                "ChatAI00._stream called from async context. Use astream() instead."
            )
        except RuntimeError:
            pass

        async def run_stream():
            accumulated_content = ""

            def create_result(content: str) -> ChatResult:
                return ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content=content))]
                )

            async for chunk_content in stream_generator():
                accumulated_content += chunk_content

                if run_manager:
                    run_manager.on_llm_new_token(chunk_content)

                yield create_result(accumulated_content)

            is_valid, cleaned = self._validate_response(accumulated_content)
            final_content = cleaned if is_valid else accumulated_content

            yield create_result(final_content)

        for result in asyncio.run(run_stream()):
            yield result


def create_chat_ai00(
    base_url: str = "http://127.0.0.1:65530",
    model_name: str = "default",
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    max_tokens: int = 1000,
    sampler_config: Optional[Dict[str, Any]] = None,
    stop: Optional[List[str]] = None,
) -> ChatAI00:
    """
    Factory function to create a ChatAI00 instance with custom sampler.

    Example:
        # Standard usage
        llm = create_chat_ai00(temperature=0.8)

        # With custom sampler (from SamplerManager)
        llm = create_chat_ai00(
            sampler_config={
                "type": "Mirostat",
                "tau": 0.5,
                "rate": 0.09
            }
        )
    """
    return ChatAI00(
        base_url=base_url,
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        sampler_config=sampler_config,
        stop=stop,
    )


async def check_ai00_health(base_url: str = "http://127.0.0.1:65530") -> bool:
    """Check if ai00-server is running and ready."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{base_url}/api/oai/models", timeout=5.0)
            return response.status_code == 200
    except:
        return False
