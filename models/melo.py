from melo.api import TTS


class Melo:
    def __init__(self):
        self.model = TTS(language='EN_NEWEST', device='auto')
        self.speaker_ids = self.model.hps.data.spk2id

    def generate(self, text):
        return self.model.tts_to_file(text, self.speaker_ids['EN-Newest'], None, speed=1.0)
