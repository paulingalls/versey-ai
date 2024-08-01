from aiortc import MediaStreamTrack
from av.audio.frame import AudioFrame
from av.audio.resampler import AudioResampler
from pyee.asyncio import AsyncIOEventEmitter
import numpy as np

from models.llm import LLM
from models.vad import VAD
from models.whisper import Whisper
from models.tts import generate

TARGET_SAMPLE_RATE = 16000


class Transform(MediaStreamTrack, AsyncIOEventEmitter):

    kind = "audio"

    def __init__(self, track):
        super().__init__()
        self.track = track
        self.vad = None
        self.buffer = None
        self.voice_buffer = None
        self.count = 1
        self.resampler = AudioResampler(format="s16", layout="mono", rate=TARGET_SAMPLE_RATE)
        self.vad = VAD(TARGET_SAMPLE_RATE)
        self.whisper = Whisper()
        self.sentence = ""

        @self.vad.on("voiceStart")
        async def on_voice_start(data):
            self.emit("voiceStart", data)

        @self.vad.on("voiceEnd")
        async def on_voice_start(data):
            self.emit("voiceEnd", data)

    def down_sample(self, frame: AudioFrame) -> AudioFrame:
        new_frames = self.resampler.resample(frame)
        return new_frames[0]

    def on_text(self, text: str, probability):
        self.sentence += text
        if text == "." or text == "!" or text == "?":
            self.emit("response", f"sentence: {self.sentence}")
            self.sentence = ""

    async def recv(self):
        frame: AudioFrame = await self.track.recv()
        resampled = self.down_sample(frame)
        if resampled.samples < 320:
            return resampled

        if self.buffer is None:
            self.buffer = resampled.to_ndarray()
        elif self.count < 8:
            self.count += 1
            self.buffer = np.concatenate((self.buffer, resampled.to_ndarray()), axis=1)
        else:
            voice_data = self.vad.vad(self.buffer)
            if "start" in voice_data:
                if "end" in voice_data:
                    text_from_voice = self.whisper.get_text(self.buffer)
                    self.emit("text", text_from_voice)
                    response = LLM.get_response(text_from_voice, self.on_text)
                    self.emit("response", response)
                else:
                    self.voice_buffer = self.buffer
            elif "end" in voice_data:
                text_from_voice = self.whisper.get_text(self.voice_buffer)
                self.voice_buffer = None
                self.emit("text", text_from_voice)
                response = LLM.get_response(text_from_voice, self.on_text)
                self.emit("response", response)
                # generate("/Users/paulingalls/src/versey-ai/mlx_models/bark", response, "small")
            elif self.voice_buffer is not None:
                self.voice_buffer = np.concatenate((self.voice_buffer, self.buffer), axis=1)
            self.buffer = resampled.to_ndarray()
            self.count = 1


        return resampled

