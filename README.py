# Local Real-time Speech-to-Text with Whisper

A fully local, real-time speech-to-text system using OpenAI's Whisper model. All processing happens on your machine - no cloud services or external APIs required. The system uses WebSocket streaming for local communication and includes voice activity detection for efficient processing.

## Features

- Real-time speech-to-text using Whisper
- Voice Activity Detection (VAD) for intelligent speech detection
- WebSocket-based streaming
- Concurrent processing of audio and transcriptions
- Environment-based configuration
- Asyncio-based architecture
- Support for multiple simultaneous clients

## Prerequisites

- Python 3.8+
- [uv](https://github.com/astral-sh/uv) for dependency management
- A microphone for audio input

## Installation

1. Clone the repository:
```bash
git clone git@github.com:bernardo-sb/sstrw.git
cd sstrw
```

2. Create a virtual environment and install dependencies using uv:
```bash
uv venv
source .venv/bin/activate  # On Unix
# or
.venv\Scripts\activate  # On Windows

uv pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your configuration:
```bash
# Client Configuration
RATE=16000
CHUNK_DURATION_MS=30  # Duration of each frame for VAD
CHANNELS=1
RECORD_SECONDS=3
```

## Usage

1. Start the server:
```bash
python src/server.py
````
or

```bash
uvicorn src.server.main:app --host 0.0.0.0 --port 8000 --reload
```

2. In a separate terminal, start the client:
```bash
python src/client.py
```

3. Start speaking - the system will:
   - Detect when you start speaking
   - Record for 3 seconds (configurable)
   - Send the audio to the server
   - Display the transcription in real-time

## Configuration

### Client Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| RATE | Audio sampling rate | 16000 |
| CHUNK_DURATION_MS | Duration of each VAD frame in milliseconds | 30 |
| CHANNELS | Number of audio channels | 1 |
| RECORD_SECONDS | Duration of each recording segment | 3 |

## API Documentation

### WebSocket Endpoints

#### `/ws/{client_id}`
- Handles WebSocket connections for real-time audio streaming
- Expects base64 encoded audio data
- Returns JSON with transcription results

Message Format:
```json
{
    "type": "audio",
    "data": "<base64-encoded-audio>",
    "duration": 3
}
```

Response Format:
```json
{
    "text": "<transcribed-text>",
    "timestamp": "<iso-format-timestamp>"
}
```

## Dependencies

Install using uv

## Common Issues

1. PyAudio installation fails:
   - macOS: `brew install portaudio`

2. WebRTC VAD installation:
   - Ubuntu/Debian: `sudo apt-get install python3-webrtcvad`
   - Other platforms: May need to install from source

3. Whisper model download:
   - The first time you run the server, it will download the Whisper model
   - Ensure you have enough disk space and a stable internet connection

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License

## Credits

- [OpenAI Whisper](https://github.com/openai/whisper)
- [WebRTC VAD](https://github.com/wiseman/py-webrtcvad)
- [FastAPI](https://fastapi.tiangolo.com/)
- [uv](https://github.com/astral-sh/uv)