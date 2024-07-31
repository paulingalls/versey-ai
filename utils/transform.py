from aiortc import MediaStreamTrack
from av.audio.frame import AudioFrame
from av.audio.resampler import AudioResampler
from pyee.asyncio import AsyncIOEventEmitter
from scipy import signal
import numpy as np

from models.vad import VAD

TARGET_SAMPLE_RATE = 16000


# def down_sample(frame: AudioFrame):
#     ratio = int(frame.sample_rate / TARGET_SAMPLE_RATE)
#     return signal.decimate(frame.to_ndarray(), ratio)

# def down_sample(frame: AudioFrame):
#     step = frame.sample_rate // 16000
#     data = frame.to_ndarray()
#     return np.array([data[0][::step]])

class Transform(MediaStreamTrack, AsyncIOEventEmitter):

    kind = "audio"

    def __init__(self, track):
        super().__init__()
        self.track = track
        self.vad = None
        self.buffer = None
        self.count = 1
        self.resampler = AudioResampler(format="s16", layout="mono", rate=TARGET_SAMPLE_RATE)

    def down_sample(self, frame: AudioFrame) -> AudioFrame:
        new_frames = self.resampler.resample(frame)
        return new_frames[0]

    async def recv(self):
        frame: AudioFrame = await self.track.recv()
        resampled = self.down_sample(frame)
        if self.vad is None:
            self.vad = VAD(TARGET_SAMPLE_RATE)

            @self.vad.on("voiceStart")
            async def on_voice_start(data):
                self.emit("voiceStart", data)

            @self.vad.on("voiceEnd")
            async def on_voice_start(data):
                self.emit("voiceEnd", data)

        if resampled.samples < 320:
            return resampled

        if self.buffer is None:
            self.buffer = resampled.to_ndarray()
        elif self.count < 8:
            self.count += 1
            self.buffer = np.concatenate((self.buffer, resampled.to_ndarray()), axis=1)
        else:
            self.vad.vad(self.buffer)
            self.buffer = resampled.to_ndarray()
            self.count = 1
        return resampled

