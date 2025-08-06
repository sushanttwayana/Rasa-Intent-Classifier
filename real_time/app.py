from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import asyncio
import json
import base64
import numpy as np
import logging
from typing import Dict, List
from datetime import datetime
import os
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Real-time Speech Transcription API", version="3.0.0")

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development, restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory for serving HTML, CSS, JS
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables for model and device
MODEL_LOADED = False
PIPE = None  # Hugging Face pipeline for ASR
DEVICE = None
SAMPLE_RATE = 16000  # Fixed sample rate for Whisper models


class ConnectionManager:
    """Manages active WebSocket connections, audio buffers, and transcription tasks."""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}  # client_id: WebSocket
        self.audio_buffers: Dict[str, List[np.ndarray]] = {}  # Stores accumulated audio chunks (numpy float32)
        self.processing_tasks: Dict[str, asyncio.Task] = {}  # Holds the async task for interim transcription
        self.is_recording: Dict[str, bool] = {}  # Tracks if a client is currently recording
        self.transcribed_segments: Dict[str, List[str]] = {}  # For appending interim transcription

    async def connect(self, websocket: WebSocket, client_id: str):
        """Accepts a new WebSocket connection."""
        await websocket.accept()
        # If client already connected, close the old connection to avoid duplicates
        if client_id in self.active_connections:
            logger.warning(f"Client {client_id} already connected, closing old connection.")
            await self.active_connections[client_id].close()

        self.active_connections[client_id] = websocket
        self.audio_buffers[client_id] = []
        self.is_recording[client_id] = False  # Initialize recording state
        self.transcribed_segments[client_id] = []
        logger.info(f"Client {client_id} connected.")
        await self.send_personal_message(json.dumps({"type": "status", "message": "Connected"}), client_id)

    def disconnect(self, client_id: str):
        """Manages client disconnection, cleans up resources."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.audio_buffers:
            del self.audio_buffers[client_id]
        if client_id in self.processing_tasks and not self.processing_tasks[client_id].done():
            self.processing_tasks[client_id].cancel()
            del self.processing_tasks[client_id]
        if client_id in self.is_recording:
            del self.is_recording[client_id]
        if client_id in self.transcribed_segments:
            del self.transcribed_segments[client_id]
        logger.info(f"Client {client_id} disconnected.")

    async def send_personal_message(self, message: str, client_id: str):
        """Sends a JSON message to a specific client."""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(message)
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}. Disconnecting client.")
                self.disconnect(client_id)  # Disconnect if message sending fails


manager = ConnectionManager()


@app.on_event("startup")
async def load_model():
    global MODEL_LOADED, PIPE, DEVICE
    try:
        logger.info("Loading speech recognition model...")
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Device set to use: {DEVICE}")

        # model_id = "openai/whisper-medium"
        model_id = "openai/whisper-large-v3-turbo"
        torch_dtype = torch.float16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )

        processor = AutoProcessor.from_pretrained(model_id)
        PIPE = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=256,
            # chunk_length_s=8,
            chunk_length_s=4,
            batch_size=1,
            torch_dtype=torch_dtype,
            device=DEVICE,
            # temperature=0  # Uncomment if supported for deterministic output
        )

        MODEL_LOADED = True
        logger.info("Model loaded successfully and ready.")
    except Exception as e:
        logger.exception("Error loading model during startup:")
        MODEL_LOADED = False


async def transcribe_segment(audio_np_segment: np.ndarray) -> str:
    """
    Transcribes a given numpy audio segment (float32) using the loaded pipeline.
    """
    if not MODEL_LOADED:
        logger.error("Attempted transcription when model is not loaded.")
        raise RuntimeError("Speech recognition model not loaded.")

    try:
        result = PIPE(audio_np_segment)
        return result["text"].strip()
    except Exception as e:
        logger.exception("Error during transcription process:")
        return ""  # Return empty string to prevent client-side errors


async def process_audio_for_client(client_id: str):
    """
    Processes audio buffer for a client using fixed-size overlapping windows,
    sends interim transcriptions, and stores them in the manager,
    deduplicating near-identical segments.
    """
    buffer_seconds = 4  # Window size in seconds
    stride_seconds = 0.5  # Sliding window stride in seconds

    buffer_samples = SAMPLE_RATE * buffer_seconds
    stride_samples = SAMPLE_RATE * stride_seconds
    last_pos = 0

    # Ensure list exists
    if client_id not in manager.transcribed_segments:
        manager.transcribed_segments[client_id] = []

    while manager.is_recording.get(client_id, False):
        try:
            await asyncio.sleep(stride_seconds)

            if client_id not in manager.audio_buffers or not manager.audio_buffers[client_id]:
                continue

            current_audio_segment = np.concatenate(manager.audio_buffers[client_id])
            total_length = len(current_audio_segment)

            # Wait until there's enough new audio to process
            if total_length < last_pos + buffer_samples:
                continue

            segment = current_audio_segment[last_pos:last_pos + buffer_samples]
            if len(segment) < SAMPLE_RATE:  # Skip very short segments < 1 sec
                continue

            transcription = await transcribe_segment(segment)
            segs = manager.transcribed_segments[client_id]
            # Only append if not similar to previous (last) segment
            if transcription and (not segs or not is_similar(transcription, segs[-1])):
                await manager.send_personal_message(
                    json.dumps({"type": "transcription", "text": transcription, "is_final": False}),
                    client_id
                )
                segs.append(transcription)
                logger.info(f"[{client_id}] Appended interim transcription: '{transcription}' (Total segments: {len(segs)})")

            last_pos += stride_samples
        except asyncio.CancelledError:
            logger.info(f"Interim processing task for {client_id} cancelled.")
            break
        except Exception as e:
            logger.exception(f"Error in interim client processing loop for {client_id}:")
            await manager.send_personal_message(
                json.dumps({"type": "error", "message": f"Interim processing error: {str(e)}"}),
                client_id
            )
            break
        
def normalize_text(text: str) -> str:
    # Lowercase and remove punctuation for approximate matching
    return re.sub(r'[^\w\s]', '', text.lower()).strip()

def is_similar(a: str, b: str) -> bool:
    # Checks if one normalized string is a substring of the other
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)
    return a_norm in b_norm or b_norm in a_norm

def merge_transcriptions(segments: List[str]) -> str:
    if not segments:
        return ""
    merged = segments[0]
    for seg in segments[1:]:
        merged_norm = normalize_text(merged)
        seg_norm = normalize_text(seg)
        overlap_len = 0
        max_overlap = min(len(merged_norm), len(seg_norm))
        # Find max overlap on normalized strings
        for i in range(max_overlap, 0, -1):
            if merged_norm[-i:] == seg_norm[:i]:
                overlap_len = i
                break

        # Find original substring length to remove in raw seg for merging
        # To avoid cutting partial words, merge on whitespace boundaries:
        cut_pos = 0
        if overlap_len > 0:
            # Approximate index in seg for overlap_len chars
            cut_pos = len(seg)  # default to full segment
            # Try to find substring in seg that matches normalized prefix of overlap_len chars
            seg_prefix_norm = seg_norm[:overlap_len]
            # Find corresponding position in raw seg for substring seg_prefix_norm ignoring case/punct
            # Simplify: We'll just remove the overlapping start chars in raw seg equal to overlap_len approx
            cut_pos = len(seg) - overlap_len  # rough approx
            
        merged += seg[cut_pos:].lstrip()
    return merged


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time audio streaming and transcription."""
    await manager.connect(websocket, client_id)

    interim_transcription_task: asyncio.Task = None
    last_pos_per_client = {client_id: 0}  # Track last processed sample position per client

    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)

            # Heartbeat (ping)
            if data["type"] == "ping":
                await manager.send_personal_message(
                    json.dumps({"type": "pong", "timestamp": datetime.now().isoformat()}),
                    client_id
                )
                continue

            # Start recording
            if data["type"] == "start_recording":
                manager.audio_buffers[client_id] = []
                manager.is_recording[client_id] = True

                # Cancel any old interim tasks
                if interim_transcription_task and not interim_transcription_task.done():
                    interim_transcription_task.cancel()
                    try:
                        await interim_transcription_task
                    except asyncio.CancelledError:
                        pass

                # Reset last_pos
                last_pos_per_client[client_id] = 0
                # Start background interim processing task
                interim_transcription_task = asyncio.create_task(process_audio_for_client(client_id))

                await manager.send_personal_message(
                    json.dumps({"type": "status", "message": "Recording started. Speak your command!"}),
                    client_id
                )
                continue

            # Stop recording
            if data["type"] == "stop_recording":
                manager.is_recording[client_id] = False

                # Cancel interim processing task if running
                if interim_transcription_task and not interim_transcription_task.done():
                    interim_transcription_task.cancel()
                    try:
                        await interim_transcription_task
                    except asyncio.CancelledError:
                        pass
                    interim_transcription_task = None

                # Process remaining unprocessed audio tail if any
                if client_id in manager.audio_buffers:
                    current_audio_segment = np.concatenate(manager.audio_buffers[client_id])
                    tail_start = last_pos_per_client.get(client_id, 0)
                    if len(current_audio_segment) > tail_start:
                        tail_segment = current_audio_segment[tail_start:]
                        if len(tail_segment) >= SAMPLE_RATE:  # At least 1 second
                            transcription = await transcribe_segment(tail_segment)
                            if transcription and (not manager.transcribed_segments.get(client_id) or transcription != (manager.transcribed_segments.get(client_id, [])[-1] if manager.transcribed_segments.get(client_id) else "")):
                                manager.transcribed_segments[client_id].append(transcription)
                                logger.info(f"[{client_id}] Final tail segment appended: '{transcription}'")

                # Merge and send final transcription
                final_segments = manager.transcribed_segments.get(client_id, [])
                logger.info(f"[{client_id}] Sending final segments count: {len(final_segments)}")
                final_transcription = merge_transcriptions(final_segments)

                if not final_transcription.strip():
                    final_transcription = "No speech detected."

                await manager.send_personal_message(
                    json.dumps({"type": "transcription", "text": final_transcription, "is_final": True}),
                    client_id
                )

                # Clear buffers
                manager.audio_buffers[client_id] = []
                manager.transcribed_segments[client_id] = []
                last_pos_per_client[client_id] = 0

                await manager.send_personal_message(
                    json.dumps({"type": "status", "message": "Recording stopped."}),
                    client_id
                )
                continue

            # Handle audio chunk
            if data["type"] == "audio_chunk":
                if not manager.is_recording.get(client_id, False):
                    logger.warning(f"Received audio chunk for {client_id} but recording is not active.")
                    continue

                try:
                    audio_bytes = base64.b64decode(data["audio"])
                    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    manager.audio_buffers[client_id].append(audio_np)
                except Exception as e:
                    logger.exception(f"Error receiving or converting audio chunk for {client_id}:")
                    await manager.send_personal_message(
                        json.dumps({"type": "error", "message": f"Audio processing error: {str(e)}"}),
                        client_id
                    )

    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected via WebSocketDisconnect.")
        manager.is_recording[client_id] = False
        if interim_transcription_task and not interim_transcription_task.done():
            interim_transcription_task.cancel()
            try:
                await interim_transcription_task
            except asyncio.CancelledError:
                pass
        manager.disconnect(client_id)
    except Exception as e:
        logger.exception(f"Unhandled WebSocket error for {client_id}:")
        manager.is_recording[client_id] = False
        if interim_transcription_task and not interim_transcription_task.done():
            interim_transcription_task.cancel()
            try:
                await interim_transcription_task
            except asyncio.CancelledError:
                pass
        manager.disconnect(client_id)


@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serves the main HTML page."""
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Error: index.html not found in static directory</h1>",
            status_code=404
        )


@app.get("/test", response_class=HTMLResponse)
async def get_test_page():
    """Serves the HTML test page for backward compatibility."""
    return await get_index()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": MODEL_LOADED,
        "device": str(DEVICE) if DEVICE else "not set"
    }


if __name__ == "__main__":
    import uvicorn
    # Run the FastAPI application using Uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
