from queue import Queue
from aiortc import MediaStreamTrack
from av.audio.frame import AudioFrame
from av.utils import Fraction
from pyee.asyncio import AsyncIOEventEmitter
import numpy as np

from utils.ai_thread import AIThread

OUTPUT_SAMPLE_RATE = 44100


class Transform(MediaStreamTrack, AsyncIOEventEmitter):
    kind = "audio"

    def __init__(self, track):
        super().__init__()
        self.track = track
        self.response_buffer = None
        self.audio_write_queue = Queue(0)
        self.audio_read_queue = Queue(0)
        self.user_text_read_queue = Queue(0)
        self.llm_text_read_queue = Queue(0)
        self.ai_thread = AIThread(self.audio_write_queue,
                                  self.audio_read_queue,
                                  self.user_text_read_queue,
                                  self.llm_text_read_queue)
        self.ai_thread.start()

    @staticmethod
    def get_silent_frame(samples, sample_rate, pts) -> AudioFrame:
        silence = np.zeros((1, samples), dtype='int16')
        new_frame = AudioFrame.from_ndarray(silence, 's16', layout="mono")
        new_frame.sample_rate = OUTPUT_SAMPLE_RATE
        new_frame.time_base = Fraction(1, OUTPUT_SAMPLE_RATE)
        new_frame.pts = round(pts * OUTPUT_SAMPLE_RATE / sample_rate)
        return new_frame

    def pull_audio_from_queue(self):
        while self.audio_read_queue.qsize() > 0:
            audio_data = self.audio_read_queue.get()
            if self.response_buffer is None:
                self.response_buffer = audio_data
            else:
                self.response_buffer = np.concatenate((self.response_buffer, audio_data), axis=0)
            self.audio_read_queue.task_done()

    def pull_text_from_queue(self):
        while self.user_text_read_queue.qsize() > 0:
            text_data = self.user_text_read_queue.get()
            self.emit("text", text_data)
            self.user_text_read_queue.task_done()
        while self.llm_text_read_queue.qsize() > 0:
            llm_data = self.llm_text_read_queue.get()
            self.emit("response", llm_data)
            self.llm_text_read_queue.task_done()

    def get_next_frame(self, samples, sample_rate, pts) -> AudioFrame:
        next_samples = None
        if self.response_buffer is None:
            return self.get_silent_frame(samples, sample_rate, pts)
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
        next_frame.pts = round(pts * OUTPUT_SAMPLE_RATE / sample_rate)
        return next_frame

    async def recv(self):
        frame: AudioFrame = await self.track.recv()
        self.audio_write_queue.put_nowait(frame)
        self.pull_audio_from_queue()
        self.pull_text_from_queue()
        output_samples = round(frame.samples * OUTPUT_SAMPLE_RATE / frame.sample_rate)
        return self.get_next_frame(output_samples, frame.sample_rate, frame.pts)
