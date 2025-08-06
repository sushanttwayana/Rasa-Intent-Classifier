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
import uuid
import os

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
SAMPLE_RATE = 16000 # Fixed sample rate for Whisper models

class ConnectionManager:
    """Manages active WebSocket connections, audio buffers, and transcription tasks."""
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}  # client_id: WebSocket
        self.audio_buffers: Dict[str, List[np.ndarray]] = {} # Stores accumulated audio chunks (numpy float32)
        self.processing_tasks: Dict[str, asyncio.Task] = {} # Holds the async task for interim transcription
        self.is_recording: Dict[str, bool] = {} # Tracks if a client is currently recording

    async def connect(self, websocket: WebSocket, client_id: str):
        """Accepts a new WebSocket connection."""
        await websocket.accept()
        # If client already connected, close the old connection to avoid duplicates
        if client_id in self.active_connections:
            logger.warning(f"Client {client_id} already connected, closing old connection.")
            await self.active_connections[client_id].close()
        
        self.active_connections[client_id] = websocket
        self.audio_buffers[client_id] = []
        self.is_recording[client_id] = False # Initialize recording state
        logger.info(f"Client {client_id} connected.")
        await self.send_personal_message(json.dumps({"type": "status", "message": "Connected"}), client_id)

    def disconnect(self, client_id: str):
        """Manages client disconnection, cleans up resources."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.audio_buffers:
            del self.audio_buffers[client_id]
        # Cancel any ongoing transcription task for this client
        if client_id in self.processing_tasks and not self.processing_tasks[client_id].done():
            self.processing_tasks[client_id].cancel()
            del self.processing_tasks[client_id]
        if client_id in self.is_recording:
            del self.is_recording[client_id]
        logger.info(f"Client {client_id} disconnected.")

    async def send_personal_message(self, message: str, client_id: str):
        """Sends a JSON message to a specific client."""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(message)
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}. Disconnecting client.")
                self.disconnect(client_id) # Disconnect if message sending fails

manager = ConnectionManager()

@app.on_event("startup")
async def load_model():
    """Loads the Distil-Whisper model on application startup."""
    global MODEL_LOADED, PIPE, DEVICE
    try:
        logger.info("Loading speech recognition model...")
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Device set to use: {DEVICE}")
        
        model_id = "distil-whisper/distil-large-v3"
        # Use bfloat16 for CUDA if supported, as it's generally faster and saves memory
        # Fallback to float32 for CPU or older GPUs
        torch_dtype = torch.float16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        model.to(DEVICE) # Move model to the selected device
        
        processor = AutoProcessor.from_pretrained(model_id)
        
        # Initialize the Hugging Face pipeline
        PIPE = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128, # Max tokens for a single command/transcription chunk
            chunk_length_s=30,  # Internal chunking for handling long audio by the pipeline
            batch_size=8,       # Can be tuned for throughput based on server resources
            torch_dtype=torch_dtype,
            device=DEVICE,
        )
        
        MODEL_LOADED = True
        logger.info(f"Model loaded successfully on device: {DEVICE}")
        
    except Exception as e:
        logger.exception("Error loading model during startup:") # Use exception for full traceback
        MODEL_LOADED = False # Ensure flag is false if loading fails

async def transcribe_segment(audio_np_segment: np.ndarray) -> str:
    """
    Transcribes a given numpy audio segment (float32) using the loaded pipeline.
    """
    if not MODEL_LOADED:
        logger.error("Attempted transcription when model is not loaded.")
        raise RuntimeError("Speech recognition model not loaded.")

    try:
        # For Distil-Whisper, we don't use initial_prompt as it's not supported
        # Instead, we rely on the model's inherent ability to recognize speech
        result = PIPE(audio_np_segment)
        
        return result["text"].strip()
        
    except Exception as e:
        logger.exception("Error during transcription process:") # Log full traceback
        return "" # Return empty string to prevent client-side errors

async def process_audio_for_client(client_id: str):
    """
    Periodically processes the accumulated audio for a client and sends interim transcriptions.
    This task runs in the background as long as the client is recording.
    """
    previous_transcription = "" # Stores the last sent interim transcription
    
    # Loop as long as the client is marked as recording
    while manager.is_recording.get(client_id, False):
        try:
            # Wait for a short interval before checking/processing audio
            await asyncio.sleep(1.0) # Process accumulated audio every 1 second for interim results

            # If no audio or client disconnected, continue loop
            if client_id not in manager.audio_buffers or not manager.audio_buffers[client_id]:
                continue
            
            # Combine all audio chunks received so far for this client
            # Note: This means re-transcribing the whole buffer each time for interim.
            # For very long utterances, a more advanced sliding window or streaming ASR
            # with internal state management would be needed. For short commands, this is acceptable.
            current_audio_segment = np.concatenate(manager.audio_buffers[client_id])
            
            # Only transcribe if there's enough audio (at least 0.5 seconds)
            if len(current_audio_segment) < SAMPLE_RATE * 0.5:
                continue
            
            transcription = await transcribe_segment(current_audio_segment)
            
            # Send interim result only if it's new and different from the last sent interim
            # This avoids flooding the client with identical updates
            if transcription and transcription != previous_transcription:
                await manager.send_personal_message(
                    json.dumps({"type": "transcription", "text": transcription, "is_final": False}),
                    client_id
                )
                previous_transcription = transcription # Update last sent transcription
                
        except asyncio.CancelledError:
            logger.info(f"Interim processing task for {client_id} cancelled.")
            break # Exit the loop if the task is cancelled
        except Exception as e:
            logger.exception(f"Error in interim client processing loop for {client_id}:")
            # Send an error message to the client
            await manager.send_personal_message(
                json.dumps({"type": "error", "message": f"Interim processing error: {str(e)}"}),
                client_id
            )
            break # Exit loop on unhandled errors

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time audio streaming and transcription."""
    await manager.connect(websocket, client_id)
    
    interim_transcription_task: asyncio.Task = None # To hold the reference to the interim task

    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            
            # Handle heartbeat messages to keep connection alive
            if data["type"] == "ping":
                await manager.send_personal_message(
                    json.dumps({"type": "pong", "timestamp": datetime.now().isoformat()}),
                    client_id
                )
                continue
            
            # Handle start recording command from client
            if data["type"] == "start_recording":
                manager.audio_buffers[client_id] = [] # Clear buffer for a new recording session
                manager.is_recording[client_id] = True # Set recording state to true
                
                # Cancel any old interim task before starting a new one
                if interim_transcription_task and not interim_transcription_task.done():
                    interim_transcription_task.cancel()
                    await interim_transcription_task # Wait for cancellation to complete
                
                # Start the background task for continuous interim transcription
                interim_transcription_task = asyncio.create_task(process_audio_for_client(client_id))

                await manager.send_personal_message(
                    json.dumps({"type": "status", "message": "Recording started. Speak your command!"}),
                    client_id
                )
                continue
                
            # Handle stop recording command from client
            if data["type"] == "stop_recording":
                manager.is_recording[client_id] = False # Stop the interim processing loop
                
                # Cancel the interim processing task if it's still running
                if interim_transcription_task and not interim_transcription_task.done():
                    interim_transcription_task.cancel()
                    try:
                        await interim_transcription_task # Await cancellation to ensure it stops cleanly
                    except asyncio.CancelledError:
                        pass # Expected if task was cancelled
                    interim_transcription_task = None # Reset the task reference

                # Process all accumulated audio for the final transcription result
                if manager.audio_buffers[client_id]:
                    full_audio = np.concatenate(manager.audio_buffers[client_id])
                    final_transcription = await transcribe_segment(full_audio)
                    await manager.send_personal_message(
                        json.dumps({"type": "transcription", "text": final_transcription, "is_final": True}),
                        client_id
                    )
                    manager.audio_buffers[client_id] = [] # Clear buffer after final processing
                else:
                    await manager.send_personal_message(
                        json.dumps({"type": "transcription", "text": "No speech detected.", "is_final": True}),
                        client_id
                    )
                await manager.send_personal_message(
                    json.dumps({"type": "status", "message": "Recording stopped."}),
                    client_id
                )
                continue

            # Handle incoming audio chunks from the client
            if data["type"] == "audio_chunk":
                if not manager.is_recording.get(client_id, False):
                    # Ignore audio if recording hasn't officially started
                    logger.warning(f"Received audio chunk for {client_id} but recording is not active.")
                    continue

                try:
                    # Decode base64 audio and convert to numpy float32 array
                    audio_bytes = base64.b64decode(data["audio"])
                    # Convert Int16 bytes to normalized Float32 numpy array
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
        manager.is_recording[client_id] = False # Ensure recording state is false
        # Cancel interim task on disconnect
        if interim_transcription_task and not interim_transcription_task.done():
            interim_transcription_task.cancel()
            try:
                await interim_transcription_task
            except asyncio.CancelledError:
                pass # Expected if task was cancelled
        manager.disconnect(client_id)
    except Exception as e:
        logger.exception(f"Unhandled WebSocket error for {client_id}:")
        manager.is_recording[client_id] = False
        # Cancel interim task on unhandled error
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