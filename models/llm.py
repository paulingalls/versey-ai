from datetime import date

from pyee.asyncio import AsyncIOEventEmitter
from mlx_lm import load, generate

# model, tokenizer = load("mlx-community/Meta-Llama-3.1-8B-8bit")
model, tokenizer = load("mlx-community/Meta-Llama-3.1-8B-Instruct-8bit")
# model, tokenizer = load("mlx-community/Meta-Llama-3.1-70B-8bit")


class LLM(AsyncIOEventEmitter):
    def __init__(self):
        super().__init__()
        self.messages = []
        self.messages.append({"role": "system",
                              "message": "A chat between a curious user and an artificial intelligence assistant.\n"
                                         "Cutting Knowledge Date: December 2023\n"
                                         f"Today's Date: {date.today().strftime("%B %d, %Y")}\n"})

    def get_prompt(self, text):
        self.messages.append({"role": "user", "message": text})
        prompt = "<|begin_of_text|>"
        for message in self.messages:
            prompt += f"\n<|start_header_id|>{message["role"]}<|end_header_id|>\n\n{message["message"]}<|eot_id|>"
        prompt += "\n<|start_header_id|>assistant<|end_header_id|>\n\n"
        return prompt

    def get_response(self, text, stream_callback):
        prompt = self.get_prompt(text)
        response = generate(model, tokenizer, prompt=prompt, verbose=True, formatter=stream_callback)
        self.messages.append({"role": "assistant", "message": response})
        print(f"response: {response}")
        return response
