from aiortc import MediaStreamTrack
from av.audio.frame import AudioFrame
from av.audio.resampler import AudioResampler
from av.utils import Fraction
from pyee.asyncio import AsyncIOEventEmitter
import numpy as np

from models.llm import LLM
from models.vad import VAD
from models.whisper import Whisper
from models.melo import Melo

MODEL_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 44100


class Transform(MediaStreamTrack, AsyncIOEventEmitter):
    kind = "audio"

    def __init__(self, track):
        super().__init__()
        self.track = track
        self.vad = None
        self.buffer = None
        self.voice_buffer = None
        self.count = 1
        self.resampler = AudioResampler(format="s16", layout="mono", rate=MODEL_SAMPLE_RATE)
        self.vad = VAD(MODEL_SAMPLE_RATE)
        self.whisper = Whisper()
        self.melo = Melo()
        self.sentence = ""
        self.response_buffer = None

        @self.vad.on("voiceStart")
        async def on_voice_start(data):
            self.emit("voiceStart", data)

        @self.vad.on("voiceEnd")
        async def on_voice_start(data):
            self.emit("voiceEnd", data)

    def down_sample(self, frame: AudioFrame) -> AudioFrame:
        new_frames = self.resampler.resample(frame)
        return new_frames[0]

    def on_text(self, text: str, _):
        self.sentence += text
        if text == "." or text == "!" or text == "?":
            self.emit("response", f"sentence: {self.sentence}")
            sentence_audio = self.melo.generate(self.sentence)
            if self.response_buffer is None:
                self.response_buffer = sentence_audio
            else:
                self.response_buffer = np.concatenate((self.response_buffer, sentence_audio), axis=0)
            self.sentence = ""

    @staticmethod
    def get_silent_frame(samples, pts) -> AudioFrame:
        silence = np.zeros((1, samples), dtype='int16')
        new_frame = AudioFrame.from_ndarray(silence, 's16', layout="mono")
        new_frame.sample_rate = OUTPUT_SAMPLE_RATE
        new_frame.time_base = Fraction(1, OUTPUT_SAMPLE_RATE)
        new_frame.pts = round(pts * OUTPUT_SAMPLE_RATE / MODEL_SAMPLE_RATE)
        return new_frame

    def get_next_frame(self, samples, pts) -> AudioFrame:
        next_samples = None
        if self.response_buffer is None:
            return self.get_silent_frame(samples, pts)
        elif len(self.response_buffer) < samples:
            next_samples = self.response_buffer * 32768
            next_samples = np.concatenate((next_samples,
                                           np.zeros((samples - len(next_samples)), dtype=np.float32)),
                                          axis=0)
            self.response_buffer = None
        else:
            next_samples = self.response_buffer[0:samples] * 32768
            self.response_buffer = self.response_buffer[samples:]

        next_frame = AudioFrame.from_ndarray(np.array([np.asarray(next_samples, dtype=np.int16)]), 's16', layout="mono")
        next_frame.sample_rate = OUTPUT_SAMPLE_RATE
        next_frame.time_base = Fraction(1, OUTPUT_SAMPLE_RATE)
        next_frame.pts = round(pts * OUTPUT_SAMPLE_RATE / MODEL_SAMPLE_RATE)
        return next_frame

    async def recv(self):
        frame: AudioFrame = await self.track.recv()
        resampled = self.down_sample(frame)
        if resampled.samples < 320:
            output_samples = round(resampled.samples * OUTPUT_SAMPLE_RATE / MODEL_SAMPLE_RATE)
            return self.get_silent_frame(output_samples, resampled.pts)

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
            elif self.voice_buffer is not None:
                self.voice_buffer = np.concatenate((self.voice_buffer, self.buffer), axis=1)
            self.buffer = resampled.to_ndarray()
            self.count = 1

        output_samples = round(resampled.samples * OUTPUT_SAMPLE_RATE / MODEL_SAMPLE_RATE)
        return self.get_next_frame(output_samples, resampled.pts)
