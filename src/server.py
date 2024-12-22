from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import List, Dict
import whisper
import numpy as np
import base64
import json
from queue import Queue
import asyncio
from datetime import datetime
import torch
import logging
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize Whisper model
"""
Size	Parameters	English-only model	Multilingual model	Required VRAM	Relative speed
tiny	39 M	tiny.en	tiny	~1 GB	~10x
base	74 M	base.en	base	~1 GB	~7x
small	244 M	small.en	small	~2 GB	~4x
medium	769 M	medium.en	medium	~5 GB	~2x
large	1550 M	N/A	large	~10 GB	1x
turbo	809 M	N/A	turbo	~6 GB	~8x
"""
WHISPER_MODEL = "turbo"
model = whisper.load_model(WHISPER_MODEL)

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.audio_buffers: Dict[str, Queue] = {}
        self.tasks: Dict[str, asyncio.Task] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.audio_buffers[client_id] = Queue()
        # Start processing task for this client
        self.tasks[client_id] = asyncio.create_task(
            self.process_audio_buffer(client_id)
        )
        logger.info(f"Client {client_id} connected")

    async def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            await websocket.close()
            del self.active_connections[client_id]
            if client_id in self.tasks:
                self.tasks[client_id].cancel()
                del self.tasks[client_id]
            if client_id in self.audio_buffers:
                del self.audio_buffers[client_id]
            logger.info(f"Client {client_id} disconnected")

    async def process_audio_buffer(self, client_id: str):
        """Process audio chunks from the buffer and run speech recognition"""
        try:
            CHUNK_DURATION = 3  # seconds
            SAMPLE_RATE = 16000
            CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION
            
            audio_buffer = np.array([], dtype=np.float32)
            
            while True:
                # Get audio chunk from queue
                if not self.audio_buffers[client_id].empty():
                    chunk = await asyncio.get_event_loop().run_in_executor(
                        None, self.audio_buffers[client_id].get
                    )
                    audio_buffer = np.append(audio_buffer, chunk)
                    
                    # Process when we have enough audio data
                    if True: #len(audio_buffer) >= CHUNK_SIZE:
                        # Process the audio chunk with Whisper
                        result = await self.transcribe_audio(audio_buffer)
                        
                        if result and result.get("text"):
                            # Send transcription back to client
                            await self.send_transcription(
                                client_id,
                                {
                                    "text": result["text"].strip(),
                                    "timestamp": datetime.now().isoformat()
                                }
                            )
                        
                        # Keep a small overlap for context
                        audio_buffer = audio_buffer[-SAMPLE_RATE:]
                
                await asyncio.sleep(0.1)  # Prevent CPU overload
                
        except asyncio.CancelledError:
            logger.info(f"Processing task cancelled for client {client_id}")
        except Exception as e:
            logger.error(f"Error processing audio for client {client_id}: {str(e)}")
            await self.disconnect(client_id)

    async def transcribe_audio(self, audio_data: np.ndarray):
        """Transcribe audio data using Whisper"""
        try:
            # Ensure audio data is in the correct format
            audio_data = audio_data.astype(np.float32)
            
            # Run transcription in a thread pool to prevent blocking
            now = datetime.now()
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: model.transcribe(
                    audio_data,
                    language="en",
                    fp16=torch.cuda.is_available()
                )
            )
            print("Total time taken:", datetime.now() - now)
            return result
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            return None

    async def send_transcription(self, client_id: str, message: dict):
        """Send transcription results back to the client"""
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(message)

manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            # Receive audio data as base64 encoded string
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                if message["type"] == "audio":
                    # Decode base64 audio data
                    audio_data = np.frombuffer(
                        base64.b64decode(message["data"]),
                        dtype=np.float32
                    )
                    # Add to processing queue
                    manager.audio_buffers[client_id].put(audio_data)
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON received from client {client_id}")
            except Exception as e:
                logger.error(f"Error processing message from client {client_id}: {str(e)}")
                
    except WebSocketDisconnect:
        await manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {str(e)}")
        await manager.disconnect(client_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)