import threading
from queue import Queue
import numpy as np
from av.audio.frame import AudioFrame
from av.audio.resampler import AudioResampler
from models.llm import LLM
from models.vad import VAD
from models.whisper import Whisper
from utils.tts_thread import TTSThread

MODEL_SAMPLE_RATE = 16000


class AIThread(threading.Thread):
    def __init__(self,
                 audio_input_queue: Queue,
                 audio_output_queue: Queue,
                 user_text_output_queue: Queue,
                 llm_text_output_queue: Queue):
        super().__init__(name='AIThread')
        self.audio_input_queue = audio_input_queue
        self.audio_output_queue = audio_output_queue
        self.user_text_output_queue = user_text_output_queue
        self.llm_text_output_queue = llm_text_output_queue
        self.sentence_queue = Queue(0)
        self.daemon = True
        self.buffer = None
        self.voice_buffer = None
        self.count = 1
        self.sentence = ""
        self.resampler = AudioResampler(format="s16", layout="mono", rate=MODEL_SAMPLE_RATE)
        self.vad = VAD(MODEL_SAMPLE_RATE)
        self.whisper = Whisper()
        self.llm = LLM()
        self.tts_thread = TTSThread(self.sentence_queue, self.audio_output_queue)
        self.tts_thread.start()

    def down_sample(self, frame: AudioFrame) -> AudioFrame:
        new_frames = self.resampler.resample(frame)
        return new_frames[0]

    def on_text(self, text: str, _):
        self.sentence += text
        if text == "." or text == "!" or text == "?" or text == "\n" or text == ":" or text.endswith(".\""):
            print(f"sentence: {self.sentence}")
            self.sentence_queue.put(self.sentence)
            self.sentence = ""

    def start_llm_response(self, buffer):
        text_from_voice = self.whisper.get_text(buffer)
        print(f"text_from_voice: {text_from_voice}")
        if len(text_from_voice.strip()) > 3:
            self.user_text_output_queue.put_nowait(text_from_voice)
            response = self.llm.get_response(text_from_voice, self.on_text)
            self.llm_text_output_queue.put_nowait(response)
        if len(self.sentence) > 0:
            print(f"last part of sentence: {self.sentence}")
            self.sentence_queue.put(self.sentence)
            self.sentence = ""

    def run(self):
        while True:
            frame: AudioFrame = self.audio_input_queue.get()
            resampled = self.down_sample(frame)
            if resampled.samples < 320:
                self.audio_input_queue.task_done()
                continue
            if self.buffer is None:
                self.buffer = resampled.to_ndarray()
            elif self.count < 8:
                self.count += 1
                self.buffer = np.concatenate((self.buffer, resampled.to_ndarray()), axis=1)
            else:
                voice_data = self.vad.vad(self.buffer)
                if "start" in voice_data:
                    print(f"voice start: {voice_data["start"]}")
                    if "end" in voice_data:
                        print(f"voice quick end: {voice_data["end"]}")
                        self.start_llm_response(self.buffer)
                        self.vad.done()
                    else:
                        self.voice_buffer = self.buffer
                elif "end" in voice_data:
                    print(f"voice end: {voice_data["end"]}")
                    self.voice_buffer = np.concatenate((self.voice_buffer, self.buffer), axis=1)
                    self.start_llm_response(self.voice_buffer)
                    self.voice_buffer = None
                    self.vad.done()
                elif self.voice_buffer is not None:
                    self.voice_buffer = np.concatenate((self.voice_buffer, self.buffer), axis=1)
                self.buffer = resampled.to_ndarray()
                self.count = 1
            self.audio_input_queue.task_done()
