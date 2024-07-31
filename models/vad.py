import torch
from numpy import ndarray
from pyee.asyncio import AsyncIOEventEmitter
from silero_vad import VADIterator

model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad:v5.1',
                              model='silero_vad',
                              force_reload=True,
                              onnx=False)


class VAD(AsyncIOEventEmitter):

    def __init__(self, sampling_rate):
        super().__init__()
        self.vad_iterator = VADIterator(model, sampling_rate=sampling_rate)

    def vad(self, audio: ndarray):
        for i in range(0, audio.shape[1], 512):
            chunk = audio[0:1, i:i+512]
            result = self.vad_iterator(torch.from_numpy(chunk))
            if result is not None:
                if "start" in result:
                    self.emit("voiceStart", result["start"])
                elif "end" in result:
                    self.emit("voiceEnd", result["end"])




