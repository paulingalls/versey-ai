# Python WebRTC Server for Voice Interaction with LLM

This Python web server provides a WebRTC interface to allow users to interact with a Large Language Model (LLM) via voice. The server uses various libraries and models to handle WebRTC, voice activity detection, speech-to-text, natural language processing, and text-to-speech functionalities.

## Features

- **WebRTC Support**: Uses `aiortc` for real-time audio streaming.
- **Voice Activity Detection**: Utilizes Silero VAD to detect when the user starts and stops speaking.
- **Speech-to-Text**: Integrates Whisper for high-quality transcription of spoken words.
- **Natural Language Processing**: Implements Llama 3.1 8B Instruct for understanding and generating responses.
- **Text-to-Speech**: Uses MeloTTS to convert responses back to speech.
- **Optimized for Mac**: Employs the mlx versions of the models for optimized performance on Mac systems.

## Installation

### Prerequisites

- Python 3.8 or higher
- `pipenv` for dependency management
- Dependencies listed in `Pipfile`
- Access to mlx versions of models

### Steps

1. Clone the repository:
    ```sh
    git clone https://github.com/paulingalls/versey-ai.git
    cd versey-ai
    ```

2. Make sure pipenv is installed:
    ```sh
    pip install pipenv --user
    ```

3. Install the required packages:
    ```sh
    pipenv install
    ```

## Usage

1. Start the server:
    ```sh
    python server.py
    ```

2. Open a browser and navigate to `http://localhost:8080` to interact with the LLM via the WebRTC interface.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss any changes or enhancements.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgements

- [aiortc](https://github.com/aiortc/aiortc)
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [Whisper](https://github.com/openai/whisper)
- [Llama](https://github.com/facebookresearch/llama)
- [MeloTTS](https://github.com/myshell-ai/MeloTTS)
 

