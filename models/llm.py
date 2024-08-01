from pyee.asyncio import AsyncIOEventEmitter
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Meta-Llama-3.1-8B-Instruct-8bit")


class LLM(AsyncIOEventEmitter):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_response(text):
        response = generate(model, tokenizer, prompt=text, verbose=True)
        return response
