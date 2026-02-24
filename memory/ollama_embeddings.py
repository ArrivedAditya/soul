from langchain_ollama import OllamaEmbeddings

model_name = "mollysama/rwkv-7-g1d:0.4b"

embed = OllamaEmbeddings(model=model_name, validate_model_on_init=True)
input_text = "The meaning of life is 42"
vector = embed.embed_query(input_text)
print(vector)
