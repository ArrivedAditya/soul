"""
A module for runing the base model.
What works
1. Ollama and ai00-server is now workihg.

What doesn't works
1. Doesn't getting thinking in chunk.
"""

import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


class Base:
    def __init__(self, server_url: str, model_name: str) -> None:
        self._server_url: str = server_url
        self.model_name: str = model_name

    async def run(self):
        model = ChatOpenAI(
            base_url=self._server_url,
            model=self.model_name,
            api_key="ollama",
            streaming=True,
        )
        message = HumanMessage("User: Explain async programming in one sentence.")

        async for chunk in model.astream([message]):
            print(chunk)


if __name__ == "__main__":
    runner_name = int(input("1 for ollama, 2 for ai00-server"))
    if runner_name == 1:
        server_url = "http://localhost:11434/v1"
    else:
        server_url = "http://localhost:65530/api/oai/v1"

    model_name = "mollysama/rwkv-7-g1d:0.4b"
    base = Base(server_url, model_name)
    asyncio.run(base.run())
