from pyee.asyncio import AsyncIOEventEmitter
from lightning_whisper_mlx import LightningWhisperMLX


class Whisper(AsyncIOEventEmitter):

    def __init__(self):
        super().__init__()
        self.whisper = LightningWhisperMLX(model="distil-medium.en", batch_size=12, quant=None)

    def get_text(self, audio):
        return self.whisper.transcribe(audio[0])["text"]