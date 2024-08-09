import torch
from numpy import ndarray
from silero_vad import VADIterator

model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad:v5.1',
                              model='silero_vad',
                              force_reload=False,
                              onnx=True)


class VAD:

    def __init__(self, sampling_rate):
        super().__init__()
        self.vad_iterator = VADIterator(model,
                                        threshold=0.6,
                                        sampling_rate=sampling_rate,
                                        min_silence_duration_ms=500)

    def vad(self, audio: ndarray):
        voice_data = {}
        for i in range(0, audio.shape[1], 512):
            chunk = audio[0:1, i:i+512]
            result = self.vad_iterator(torch.from_numpy(chunk[0]))
            if result is not None:
                if "start" in result:
                    voice_data["start"] = result["start"]
                if "end" in result:
                    voice_data["end"] = result["end"]
        return voice_data

    def done(self):
        self.vad_iterator.reset_states()


