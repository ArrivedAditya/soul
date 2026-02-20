import asyncio

from base.base_model import Base

print("soul v0.3.0 - Creating Phase")

runner_name = int(input("1 for ollama, 2 for ai00-server: "))
if runner_name == 1:
    server_url = "http://localhost:11434/v1"
else:
    server_url = "http://localhost:65530/api/oai/v1"

model_name = "mollysama/rwkv-7-g1d:0.4b"
base = Base(server_url, model_name)
asyncio.run(base.run())
