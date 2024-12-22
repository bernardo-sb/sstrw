import os
import asyncio
import websockets
import json
import base64
import pyaudio
import numpy as np
import uuid
import webrtcvad
import threading
import queue
from typing import Optional
import time
from dotenv import load_dotenv


class VoiceActivatedClient:
    def __init__(
        self, rate: int, chunk_duration: int, channels: int, record_seconds: int
    ):
        # Audio configuration
        self.RATE = rate
        self.CHUNK_DURATION_MS = chunk_duration  # Duration of each frame for VAD
        self.CHUNK_SIZE = int(self.RATE * self.CHUNK_DURATION_MS / 1000)
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = channels
        self.RECORD_SECONDS = record_seconds

        # VAD configuration
        self.vad = webrtcvad.Vad(3)

        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream = None

        # Websocket
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.running = False

        # Thread-safe queue for responses
        self.response_queue = queue.Queue()

    def start_stream(self):
        self.stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK_SIZE,
        )

    def stop_stream(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        pcm_data = (audio_chunk * 32768).astype(np.int16)
        return self.vad.is_speech(pcm_data.tobytes(), self.RATE)

    async def receive_responses(self):
        """Thread for receiving and processing responses"""
        try:
            while self.running:
                try:
                    response = await self.websocket.recv()
                    result = json.loads(response)
                    print(f"Transcription: {result["text"]}")
                except websockets.exceptions.ConnectionClosed:
                    break
                except Exception as e:
                    print(f"Error receiving response: {e}")
                    break
        except Exception as e:
            print(f"Receive thread error: {e}")

    async def record_and_send(self):
        """Main thread for recording and sending audio"""
        self.start_stream()
        frames = []
        speech_detected = False
        start_time = None

        try:
            while self.running:
                # Read audio chunk
                data = self.stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.float32)

                # Check for speech
                if not speech_detected:
                    if self.is_speech(audio_chunk):
                        speech_detected = True
                        start_time = time.time()
                        frames = [audio_chunk]  # Start collecting frames
                        print("Speech detected, recording...")

                # If we"re recording, collect frames
                elif speech_detected:
                    frames.append(audio_chunk)
                    elapsed_time = time.time() - start_time

                    # Check if we"ve recorded for 3 seconds
                    if elapsed_time >= self.RECORD_SECONDS:
                        # Combine all frames
                        audio_data = np.concatenate(frames)

                        # Encode and send
                        encoded_data = base64.b64encode(audio_data.tobytes()).decode(
                            "utf-8"
                        )
                        try:
                            await self.websocket.send(
                                json.dumps(
                                    {
                                        "type": "audio",
                                        "data": encoded_data,
                                        "duration": self.RECORD_SECONDS,
                                    }
                                )
                            )
                            print("Audio chunk sent")
                        except Exception as e:
                            print(f"Error sending audio: {e}")

                        # Reset for next recording
                        speech_detected = False
                        frames = []
                        start_time = None

                # Small sleep to prevent CPU overload
                await asyncio.sleep(0.001)

        finally:
            self.stop_stream()

    async def run(self):
        """Main entry point to run the client"""
        client_id = str(uuid.uuid4())

        try:
            async with websockets.connect(f"ws://localhost:8000/ws/{client_id}") as ws:
                self.websocket = ws
                self.running = True
                print("Connected to server. Start speaking...")

                # Create tasks for sending and receiving
                receive_task = asyncio.create_task(self.receive_responses())
                send_task = asyncio.create_task(self.record_and_send())

                # Wait for both tasks
                await asyncio.gather(receive_task, send_task)

        except Exception as e:
            print(f"Connection error: {e}")
        finally:
            self.running = False
            self.stop_stream()


def main():
    load_dotenv()
    client = VoiceActivatedClient(
        os.environ.get("RATE", 1600),
        os.environ.get("CHUNK_DURATION", 30),
        os.environ.get("CHANNELS", 1),
        os.environ.get("RECORD_SECONDS", 3),
    )
    asyncio.run(client.run())


if __name__ == "__main__":
    main()
