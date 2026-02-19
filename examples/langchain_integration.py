"""
Example: Integrating ChatAI00 with existing SamplerManager.

This demonstrates how to use the LangChain wrapper while keeping
your custom sampler configurations.
"""

import asyncio
from models.langchain_ai00 import create_chat_ai00, ChatAI00
from core.sampler_manager import SamplerManager


def create_llm_from_sampler(sampler_name: str) -> ChatAI00:
    """
    Create a LangChain ChatAI00 from your existing SamplerManager.

    Args:
        sampler_name: Name of sampler preset (e.g., 'chat', 'reflect', 'task')

    Returns:
        Configured ChatAI00 instance
    """
    manager = SamplerManager()
    sampler = manager.get_sampler(sampler_name)

    if sampler is None:
        print(f"Warning: Unknown sampler '{sampler_name}', using defaults")
        return create_chat_ai00(temperature=0.8)

    return create_chat_ai00(sampler_config=sampler.to_api_dict())


async def basic_example():
    """Basic LangChain usage without your custom components."""
    from langchain_core.messages import HumanMessage

    llm = create_chat_ai00(temperature=0.8)

    response = await llm.ainvoke([HumanMessage(content="Hello, who are you?")])
    print(f"Response: {response.content}")


async def with_sampler_example():
    """Example using your SamplerManager presets."""
    from langchain_core.messages import HumanMessage

    # Use 'reflect' sampler for creative thinking
    llm = create_llm_from_sampler("reflect")

    response = await llm.ainvoke(
        [HumanMessage(content="What are your thoughts on artificial consciousness?")]
    )
    print(f"Reflect response: {response.content[:200]}...")


async def lcel_example():
    """Example using LangChain Expression Language (LCEL)."""
    from langchain_core.messages import HumanMessage
    from langchain.prompts import ChatPromptTemplate

    llm = create_chat_ai00(temperature=0.7)

    # Create a chain
    chain = (
        ChatPromptTemplate.from_messages(
            [
                ("system", "You are {name}, a thoughtful AI companion."),
                ("user", "{greeting}"),
            ]
        )
        | llm
    )

    response = chain.invoke({"name": "Soul", "greeting": "What brings you joy today?"})

    print(f"LCEL response: {response.content}")


async def streaming_example():
    """Example with streaming response."""
    from langchain_core.messages import HumanMessage

    llm = create_chat_ai00(temperature=0.8)

    print("Streaming: ", end="", flush=True)
    async for chunk in llm.astream([HumanMessage(content="Count to 5")]):
        print(chunk.content, end="", flush=True)
    print()


async def main():
    print("=== Basic Example ===")
    try:
        await basic_example()
    except Exception as e:
        print(f"(ai00-server not running: {e})")

    print("\n=== With Sampler Preset ===")
    try:
        await with_sampler_example()
    except Exception as e:
        print(f"(ai00-server not running: {e})")

    print("\n=== LCEL Chain ===")
    try:
        await lcel_example()
    except Exception as e:
        print(f"(ai00-server not running: {e})")

    print("\n=== Streaming ===")
    try:
        await streaming_example()
    except Exception as e:
        print(f"(ai00-server not running: {e})")


if __name__ == "__main__":
    asyncio.run(main())
