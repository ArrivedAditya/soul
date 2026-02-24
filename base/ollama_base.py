import asyncio
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

model_name = "mollysama/rwkv-7-g1d:0.4b"

model = ChatOllama(model=model_name, reasoning=True, validate_model_on_init=True)

message = HumanMessage("Hello!")


async def run(model, message):
    async for chunk in model.astream([message]):
        print(chunk)


asyncio.run(run(model, message))
