"""
Tests for LangChain-compatible ai00-server wrapper.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from models.langchain_ai00 import ChatAI00, create_chat_ai00, check_ai00_health
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


class TestChatAI00:
    """Test cases for ChatAI00 class."""

    def test_default_initialization(self):
        """Test default values on initialization."""
        llm = ChatAI00()

        assert llm.base_url == "http://127.0.0.1:65530"
        assert llm.model_name == "default"
        assert llm.max_tokens == 1000
        assert llm.temperature == 1.0
        assert llm._llm_type == "ai00-chat"

    def test_custom_initialization(self):
        """Test custom values on initialization."""
        llm = ChatAI00(
            base_url="http://localhost:8080",
            model_name="rwkv7-goose",
            max_tokens=500,
            temperature=0.7,
            top_p=0.9,
        )

        assert llm.base_url == "http://localhost:8080"
        assert llm.model_name == "rwkv7-goose"
        assert llm.max_tokens == 500
        assert llm.temperature == 0.7
        assert llm.top_p == 0.9

    def test_identifying_params(self):
        """Test _identifying_params property."""
        llm = ChatAI00(model_name="test-model", temperature=0.5, top_p=0.8)

        params = llm._identifying_params

        assert params["model_name"] == "test-model"
        assert params["temperature"] == 0.5
        assert params["top_p"] == 0.8
        assert params["max_tokens"] == 1000


class TestMessageConversion:
    """Test message conversion methods."""

    def test_convert_human_message(self):
        """Test converting HumanMessage."""
        llm = ChatAI00()
        msg = HumanMessage(content="Hello")

        result = llm._convert_messages([msg])

        assert result == [{"role": "user", "content": "Hello"}]

    def test_convert_ai_message(self):
        """Test converting AIMessage."""
        llm = ChatAI00()
        msg = AIMessage(content="Hi there")

        result = llm._convert_messages([msg])

        assert result == [{"role": "assistant", "content": "Hi there"}]

    def test_convert_system_message(self):
        """Test converting SystemMessage."""
        llm = ChatAI00()
        msg = SystemMessage(content="You are helpful")

        result = llm._convert_messages([msg])

        assert result == [{"role": "system", "content": "You are helpful"}]

    def test_convert_multiple_messages(self):
        """Test converting multiple messages."""
        llm = ChatAI00()
        messages = [
            SystemMessage(content="You are Soul"),
            HumanMessage(content="Hello"),
            AIMessage(content="Hi!"),
        ]

        result = llm._convert_messages(messages)

        assert len(result) == 3
        assert result[0] == {"role": "system", "content": "You are Soul"}
        assert result[1] == {"role": "user", "content": "Hello"}
        assert result[2] == {"role": "assistant", "content": "Hi!"}


class TestSamplerConfig:
    """Test sampler configuration methods."""

    def test_default_sampler_params(self):
        """Test default sampler parameters."""
        llm = ChatAI00(temperature=0.8)

        params = llm._get_sampler_params()

        assert params["type"] == "Nucleus"
        assert params["temperature"] == 0.8

    def test_custom_sampler_config(self):
        """Test custom sampler configuration."""
        llm = ChatAI00(sampler_config={"type": "Mirostat", "tau": 0.5, "rate": 0.09})

        params = llm._get_sampler_params()

        assert params["type"] == "Mirostat"
        assert params["tau"] == 0.5
        assert params["rate"] == 0.09

    def test_sampler_with_top_p(self):
        """Test sampler with top_p."""
        llm = ChatAI00(temperature=0.7, top_p=0.5)

        params = llm._get_sampler_params()

        assert params["temperature"] == 0.7
        assert params["top_p"] == 0.5


class TestResponseValidation:
    """Test response validation."""

    def test_empty_response(self):
        """Test empty response passes."""
        llm = ChatAI00()

        is_valid, cleaned = llm._validate_response("")

        assert is_valid is True
        assert cleaned == ""

    def test_valid_response(self):
        """Test valid response passes."""
        llm = ChatAI00()

        is_valid, cleaned = llm._validate_response("Hello, how can I help you?")

        assert is_valid is True
        assert cleaned == "Hello, how can I help you?"

    def test_rambling_system_prompt(self):
        """Test rambling detection for system prompt leakage."""
        llm = ChatAI00()

        response = """Hello there!
As an AI language model, I am programmed to help you.
How can I assist you today?"""

        is_valid, cleaned = llm._validate_response(response)

        assert is_valid is True
        assert "As an AI" not in cleaned
        assert "I am programmed" not in cleaned

    def test_all_rambling_filtered(self):
        """Test when all content is rambling."""
        llm = ChatAI00()

        response = "As an AI language model, I am programmed to follow instructions."

        is_valid, cleaned = llm._validate_response(response)

        assert is_valid is False
        assert "filtered" in cleaned.lower()


class TestCreateChatAI00:
    """Test factory function."""

    def test_create_with_defaults(self):
        """Test factory with default values."""
        llm = create_chat_ai00()

        assert isinstance(llm, ChatAI00)
        assert llm.temperature == 1.0

    def test_create_with_temperature(self):
        """Test factory with custom temperature."""
        llm = create_chat_ai00(temperature=0.5)

        assert llm.temperature == 0.5

    def test_create_with_sampler_config(self):
        """Test factory with sampler config."""
        config = {"type": "Typical", "temperature": 0.9, "top_p": 0.5}
        llm = create_chat_ai00(sampler_config=config)

        assert llm.sampler_config == config

    def test_create_with_stop_sequences(self):
        """Test factory with custom stop sequences."""
        stop = ["END", "STOP"]
        llm = create_chat_ai00(stop=stop)

        assert llm.stop == stop


class TestHealthCheck:
    """Test health check function."""

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test health check when server is up."""
        with patch("models.langchain_ai00.httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client.return_value.__aenter__.return_value.get.return_value = (
                mock_response
            )

            result = await check_ai00_health()

            assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test health check when server is down."""
        with patch("models.langchain_ai00.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get.side_effect = (
                Exception("Connection refused")
            )

            result = await check_ai00_health()

            assert result is False


class TestGenerate:
    """Test generate methods."""

    @pytest.mark.asyncio
    async def test_agenerate_success(self):
        """Test async generate with mock response."""
        llm = ChatAI00()

        mock_response = {"choices": [{"message": {"content": "Hello from AI!"}}]}

        with patch.object(llm, "_create_client") as mock_create:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock()
            mock_client.post.return_value.json = lambda: mock_response
            mock_client.post.return_value.raise_for_status = MagicMock()
            mock_create.return_value = mock_client

            result = await llm._agenerate([HumanMessage(content="Hi")])

            assert len(result.generations) == 1
            assert result.generations[0].message.content == "Hello from AI!"

    @pytest.mark.asyncio
    async def test_agenerate_with_sampler(self):
        """Test async generate with custom sampler."""
        llm = ChatAI00(sampler_config={"type": "Mirostat", "tau": 0.5, "rate": 0.09})

        mock_response = {"choices": [{"message": {"content": "Test response"}}]}

        with patch.object(llm, "_create_client") as mock_create:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock()
            mock_client.post.return_value.json = lambda: mock_response
            mock_client.post.return_value.raise_for_status = MagicMock()
            mock_create.return_value = mock_client

            result = await llm._agenerate([HumanMessage(content="Test")])

            # Verify sampler config was passed
            call_args = mock_client.post.call_args
            assert "extra_body" in call_args.kwargs["json"]
            assert (
                call_args.kwargs["json"]["extra_body"]["sampler_override"]["type"]
                == "Mirostat"
            )


class TestSyncGenerate:
    """Test synchronous generate."""

    @pytest.mark.asyncio
    async def test_generate_raises_in_async_context(self):
        """Test sync generate raises error in async context."""
        llm = ChatAI00()

        with pytest.raises(RuntimeError):
            llm._generate([HumanMessage(content="test")])

    def test_generate_sync(self):
        """Test sync generate works."""
        llm = ChatAI00()

        mock_response = {"choices": [{"message": {"content": "Sync response"}}]}

        with patch.object(llm, "_agenerate", return_value=mock_response) as mock_agen:
            result = llm._generate([HumanMessage(content="Hi")])

            assert result == mock_response


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
