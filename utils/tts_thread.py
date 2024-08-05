import threading
from queue import Queue
from models.melo import Melo


class TTSThread(threading.Thread):
    def __init__(self, input_queue: Queue, output_queue: Queue):
        super().__init__(name='TTSThread')
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.melo = Melo()

    def run(self):
        while True:
            sentence = self.input_queue.get()
            sentence_audio = self.melo.generate(sentence)
            self.output_queue.put(sentence_audio)
            self.input_queue.task_done()