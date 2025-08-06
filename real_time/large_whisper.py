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
from contextlib import asynccontextmanager

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables
MODEL_LOADED = False
PIPE = None  # ASR pipeline
DEVICE = None
SAMPLE_RATE = 16000  # Whisper model sample rate

client_languages: Dict[str, str] = {}

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

# ----------------------------------download the model locally

# model_id = "openai/whisper-large-v3-turbo"
# local_dir = "models/whisper-large-v3-turbo"

# AutoModelForSpeechSeq2Seq.from_pretrained(model_id, cache_dir=local_dir)
# AutoProcessor.from_pretrained(model_id, cache_dir=local_dir)
# print("Model downloaded successfully")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL_LOADED, PIPE, DEVICE
    try:
        logger.info("Loading speech recognition model...")
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {DEVICE}")

        # model_id = "openai/whisper-large-v3-turbo"
        
        local_dir = "models/whisper-large-v3-turbo/models--openai--whisper-large-v3-turbo/snapshots/41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
        # G:\Vanilla_Tech\speech2text\real_time\models\whisper-large-v3-turbo\models--openai--whisper-large-v3-turbo\snapshots\41f01f3fe87f28c78e2fbf8b568835947dd65ed9
        # model = AutoModelForSpeechSeq2Seq.from_pretrained(local_dir, ...)
        # processor = AutoProcessor.from_pretrained(local_dir, ...)

        torch_dtype = torch.float16 if DEVICE.type == 'cuda' and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            local_dir,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )

        processor = AutoProcessor.from_pretrained(local_dir)

        # Set chunk_length_s to 30 to allow max context on large model
        PIPE = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=256,
            chunk_length_s=30,
            batch_size=1,
            torch_dtype=torch_dtype,
            device=DEVICE,
        )

        MODEL_LOADED = True
        logger.info("Model loaded and pipeline ready.")
        yield

    except Exception as e:
        logger.exception("Failed to load model:")
        MODEL_LOADED = False
        yield
    finally:
        logger.info("Shutting down...")

app = FastAPI(title="Real-time Speech Transcription API", version="3.3.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.audio_buffers: Dict[str, List[np.ndarray]] = {}
        self.processing_tasks: Dict[str, asyncio.Task] = {}
        self.is_recording: Dict[str, bool] = {}
        self.transcribed_segments: Dict[str, List[str]] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        if client_id in self.active_connections:
            logger.warning(f"Client {client_id} reconnected, closing old connection.")
            await self.active_connections[client_id].close()

        self.active_connections[client_id] = websocket
        self.audio_buffers[client_id] = []
        self.is_recording[client_id] = False
        self.transcribed_segments[client_id] = []
        logger.info(f"Client {client_id} connected.")
        await self.send_personal_message(json.dumps({"type": "status", "message": "Connected"}), client_id)

    def disconnect(self, client_id: str):
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
        if client_id in client_languages:
            del client_languages[client_id]
        logger.info(f"Client {client_id} disconnected.")

    async def send_personal_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(message)
            except Exception as e:
                logger.error(f"Failed sending to {client_id}: {e} — disconnecting.")
                self.disconnect(client_id)

manager = ConnectionManager()

async def transcribe_segment(audio_np_segment: np.ndarray, language: str) -> str:
    if not MODEL_LOADED:
        logger.error("Transcription requested but model not loaded.")
        raise RuntimeError("Model not loaded.")
    try:
        result = PIPE(audio_np_segment, generate_kwargs={"language": language})
        return result["text"].strip()
    except Exception:
        logger.exception("Transcription error:")
        return ""

async def transcribe_full_audio(client_id: str, full_audio: np.ndarray, language: str) -> str:
    """Final pass over full audio with max 30s chunks."""
    MAX_CHUNK = SAMPLE_RATE * 30
    segments = []
    start = 0
    while start < len(full_audio):
        segment = full_audio[start : start + MAX_CHUNK]
        text = await transcribe_segment(segment, language)
        if text:
            segments.append(text)
        start += MAX_CHUNK
    return ' '.join(segments).strip()

async def process_audio_for_client(client_id: str, language: str):
    CHUNK_SECONDS = 15
    CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS

    audio_buffer = np.array([], dtype=np.float32)

    while manager.is_recording.get(client_id, False):
        try:
            await asyncio.sleep(0.2)
            # Append all new buffers
            while manager.audio_buffers[client_id]:
                audio_buffer = np.concatenate([audio_buffer, manager.audio_buffers[client_id].pop(0)])
            # Process all full chunks
            while len(audio_buffer) >= CHUNK_SAMPLES:
                chunk = audio_buffer[:CHUNK_SAMPLES]
                audio_buffer = audio_buffer[CHUNK_SAMPLES:]
                transcription = await transcribe_segment(chunk, language)
                if transcription:
                    await manager.send_personal_message(
                        json.dumps({"type": "transcription", "text": transcription, "is_final": False}),
                        client_id
                    )
                    manager.transcribed_segments[client_id].append(transcription)

        except asyncio.CancelledError:
            logger.info(f"Interim task for {client_id} cancelled.")
            break
        except Exception:
            logger.exception(f"Interim processing error for {client_id}")
            await manager.send_personal_message(
                json.dumps({"type": "error", "message": "Interim processing error"}),
                client_id
            )
            break

    # Final leftover chunk
    if len(audio_buffer) > 0:
        try:
            transcription = await transcribe_segment(audio_buffer, language)
            if transcription:
                await manager.send_personal_message(
                    json.dumps({"type": "transcription", "text": transcription, "is_final": False}),
                    client_id
                )
                manager.transcribed_segments[client_id].append(transcription)
        except Exception:
            logger.exception(f"Error processing final chunk for {client_id}")
            await manager.send_personal_message(
                json.dumps({"type": "error", "message": "Final chunk processing error"}),
                client_id
            )


def merge_transcriptions(segments: List[str]) -> str:
    return ' '.join(segments).strip() if segments else ""

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    interim_task: asyncio.Task = None

    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)

            if data["type"] == "ping":
                await manager.send_personal_message(
                    json.dumps({"type": "pong", "timestamp": datetime.now().isoformat()}),
                    client_id
                )
                continue

            if data["type"] == "start_recording":
                lang = data.get("language", "en")
                supported = ["en", "ms", "my", "zh", "id", "ne", "ta", "bn"]
                if lang not in supported:
                    logger.warning(f"Unsupported language {lang} from {client_id}, defaulting to en")
                    lang = "en"

                client_languages[client_id] = lang
                logger.info(f"[{client_id}] Started recording; language={lang}")

                manager.audio_buffers[client_id] = []
                manager.is_recording[client_id] = True

                if interim_task and not interim_task.done():
                    interim_task.cancel()
                    try:
                        await interim_task
                    except asyncio.CancelledError:
                        pass

                interim_task = asyncio.create_task(process_audio_for_client(client_id, lang))

                await manager.send_personal_message(
                    json.dumps({"type": "status", "message": "Recording started.", "language": lang}),
                    client_id
                )
                continue

            if data["type"] == "stop_recording":
                manager.is_recording[client_id] = False

                if interim_task and not interim_task.done():
                    interim_task.cancel()
                    try:
                        await interim_task
                    except asyncio.CancelledError:
                        pass
                    interim_task = None

                # Use only the collected interim transcriptions as the final output
                final_segments = manager.transcribed_segments.get(client_id, [])
                final_text = ' '.join(final_segments).strip() if final_segments else "No speech detected."

                await manager.send_personal_message(
                    json.dumps({"type": "transcription", "text": final_text, "is_final": True}),
                    client_id
                )

                # Reset buffers and state
                manager.audio_buffers[client_id] = []
                manager.transcribed_segments[client_id] = []

                await manager.send_personal_message(
                    json.dumps({"type": "status", "message": "Recording stopped."}),
                    client_id
                )
                continue

            if data["type"] == "audio_chunk":
                if not manager.is_recording.get(client_id, False):
                    continue

                try:
                    audio_bytes = base64.b64decode(data["audio"])
                    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    manager.audio_buffers[client_id].append(audio_np)
                except Exception:
                    logger.exception(f"Audio chunk processing error for {client_id}")
                    await manager.send_personal_message(
                        json.dumps({"type": "error", "message": "Audio chunk processing error"}),
                        client_id
                    )

    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected.")
        manager.is_recording[client_id] = False
        if interim_task and not interim_task.done():
            interim_task.cancel()
            try:
                await interim_task
            except asyncio.CancelledError:
                pass
        manager.disconnect(client_id)
    except Exception:
        logger.exception(f"Unhandled server error for {client_id}")
        manager.is_recording[client_id] = False
        if interim_task and not interim_task.done():
            interim_task.cancel()
            try:
                await interim_task
            except asyncio.CancelledError:
                pass
        manager.disconnect(client_id)

@app.get("/", response_class=HTMLResponse)
async def get_root():
    return HTMLResponse(content=open("static/index.html").read())

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": MODEL_LOADED,
        "device": str(DEVICE) if DEVICE else "not set"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



# ______________________________________________________________________________________________________________________________-


# from fastapi import FastAPI, WebSocket, WebSocketDisconnect
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import HTMLResponse
# from fastapi.staticfiles import StaticFiles
# import torch
# from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
# import asyncio
# import json
# import base64
# import numpy as np
# import logging
# from typing import Dict, List
# from datetime import datetime
# import os
# import re
# from contextlib import asynccontextmanager

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # Global variables for model and device
# MODEL_LOADED = False
# PIPE = None  # Hugging Face pipeline for ASR
# DEVICE = None
# SAMPLE_RATE = 16000  # Fixed sample rate for Whisper models

# # A dictionary to map client_id to their chosen language
# client_languages: Dict[str, str] = {}

# # The `lifespan` context manager is the recommended way to handle startup/shutdown logic.
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """
#     Handles application startup and shutdown events.
#     The code before 'yield' runs on startup. The code after 'yield' runs on shutdown.
#     """
#     global MODEL_LOADED, PIPE, DEVICE
#     try:
#         logger.info("Application startup: Loading speech recognition model...")
#         DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         logger.info(f"Device set to use: {DEVICE}")

#         model_id = "openai/whisper-large-v3-turbo"
#         torch_dtype = torch.float16 if DEVICE.type == 'cuda' and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float32

#         model = AutoModelForSpeechSeq2Seq.from_pretrained(
#             model_id,
#             torch_dtype=torch_dtype,
#             low_cpu_mem_usage=True,
#             use_safetensors=True
#         )

#         processor = AutoProcessor.from_pretrained(model_id)

#         PIPE = pipeline(
#             "automatic-speech-recognition",
#             model=model,
#             tokenizer=processor.tokenizer,
#             feature_extractor=processor.feature_extractor,
#             max_new_tokens=256,
#             chunk_length_s=4,
#             batch_size=1,
#             torch_dtype=torch_dtype,
#             device=DEVICE,
#         )

#         MODEL_LOADED = True
#         logger.info("Model loaded successfully and ready.")
#         yield  # The application can now serve requests.

#     except Exception as e:
#         logger.exception("Error loading model during startup:")
#         MODEL_LOADED = False
#         yield # The application will still start, but in an error state for this feature.
    
#     finally:
#         logger.info("Application shutdown: Cleaning up resources.")
#         # Any necessary cleanup code would go here.
#         # For this example, no explicit model unloading is required.


# # Initialize FastAPI app with the lifespan event handler
# app = FastAPI(
#     title="Real-time Speech Transcription API",
#     version="3.2.0",
#     lifespan=lifespan
# )

# # Add CORS middleware for frontend access
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Mount static files directory for serving HTML, CSS, JS
# if os.path.exists("static"):
#     app.mount("/static", StaticFiles(directory="static"), name="static")


# class ConnectionManager:
#     """Manages active WebSocket connections, audio buffers, and transcription tasks."""

#     def __init__(self):
#         self.active_connections: Dict[str, WebSocket] = {}
#         self.audio_buffers: Dict[str, List[np.ndarray]] = {}
#         self.processing_tasks: Dict[str, asyncio.Task] = {}
#         self.is_recording: Dict[str, bool] = {}
#         self.transcribed_segments: Dict[str, List[str]] = {}

#     async def connect(self, websocket: WebSocket, client_id: str):
#         """Accepts a new WebSocket connection."""
#         await websocket.accept()
#         if client_id in self.active_connections:
#             logger.warning(f"Client {client_id} already connected, closing old connection.")
#             await self.active_connections[client_id].close()

#         self.active_connections[client_id] = websocket
#         self.audio_buffers[client_id] = []
#         self.is_recording[client_id] = False
#         self.transcribed_segments[client_id] = []
#         logger.info(f"Client {client_id} connected.")
#         await self.send_personal_message(json.dumps({"type": "status", "message": "Connected"}), client_id)

#     def disconnect(self, client_id: str):
#         """Manages client disconnection, cleans up resources."""
#         if client_id in self.active_connections:
#             del self.active_connections[client_id]
#         if client_id in self.audio_buffers:
#             del self.audio_buffers[client_id]
#         if client_id in self.processing_tasks and not self.processing_tasks[client_id].done():
#             self.processing_tasks[client_id].cancel()
#             del self.processing_tasks[client_id]
#         if client_id in self.is_recording:
#             del self.is_recording[client_id]
#         if client_id in self.transcribed_segments:
#             del self.transcribed_segments[client_id]
#         if client_id in client_languages:
#             del client_languages[client_id]
#         logger.info(f"Client {client_id} disconnected.")

#     async def send_personal_message(self, message: str, client_id: str):
#         """Sends a JSON message to a specific client."""
#         if client_id in self.active_connections:
#             try:
#                 await self.active_connections[client_id].send_text(message)
#             except Exception as e:
#                 logger.error(f"Error sending message to {client_id}: {e}. Disconnecting client.")
#                 self.disconnect(client_id)

# manager = ConnectionManager()

# async def transcribe_segment(audio_np_segment: np.ndarray, language: str) -> str:
#     """
#     Transcribes a given numpy audio segment (float32) using the loaded pipeline,
#     with explicit language specification.
#     """
#     if not MODEL_LOADED:
#         logger.error("Attempted transcription when model is not loaded.")
#         raise RuntimeError("Speech recognition model not loaded.")

#     try:
#         result = PIPE(audio_np_segment, generate_kwargs={"language": language})
#         return result["text"].strip()
#     except Exception as e:
#         logger.exception("Error during transcription process:")
#         return ""

# async def process_audio_for_client(client_id: str, language: str):
#     """
#     Processes audio buffer for a client using fixed-size overlapping windows,
#     sends interim transcriptions, and stores them in the manager.
#     """
#     buffer_seconds = 4
#     stride_seconds = 1

#     buffer_samples = SAMPLE_RATE * buffer_seconds
#     stride_samples = SAMPLE_RATE * stride_seconds
#     last_pos = 0

#     if client_id not in manager.transcribed_segments:
#         manager.transcribed_segments[client_id] = []

#     while manager.is_recording.get(client_id, False):
#         try:
#             await asyncio.sleep(stride_seconds)
#             if client_id not in manager.audio_buffers or not manager.audio_buffers[client_id]:
#                 continue

#             current_audio_segment = np.concatenate(manager.audio_buffers[client_id])
#             total_length = len(current_audio_segment)
#             if total_length < last_pos + buffer_samples:
#                 continue

#             segment = current_audio_segment[last_pos:last_pos + buffer_samples]
#             if len(segment) < SAMPLE_RATE:
#                 continue

#             transcription = await transcribe_segment(segment, language)
#             segs = manager.transcribed_segments[client_id]
#             if transcription and (not segs or not is_similar(transcription, segs[-1])):
#                 await manager.send_personal_message(
#                     json.dumps({"type": "transcription", "text": transcription, "is_final": False}),
#                     client_id
#                 )
#                 segs.append(transcription)
#                 logger.info(f"[{client_id}] Appended interim transcription: '{transcription}' (Total segments: {len(segs)})")

#             last_pos += stride_samples
#         except asyncio.CancelledError:
#             logger.info(f"Interim processing task for {client_id} cancelled.")
#             break
#         except Exception as e:
#             logger.exception(f"Error in interim client processing loop for {client_id}:")
#             await manager.send_personal_message(
#                 json.dumps({"type": "error", "message": f"Interim processing error: {str(e)}"}),
#                 client_id
#             )
#             break

# def normalize_text(text: str) -> str:
#     """Helper function to normalize text for comparison."""
#     return re.sub(r'[^\w\s]', '', text.lower()).strip()

# def is_similar(a: str, b: str) -> bool:
#     """Checks if one normalized string is a substring of the other."""
#     a_norm = normalize_text(a)
#     b_norm = normalize_text(b)
#     return a_norm in b_norm or b_norm in a_norm

# def merge_transcriptions(segments: List[str]) -> str:
#     """A simplified merge function based on word overlap."""
#     if not segments:
#         return ""
    
#     merged_text = segments[0]
#     for next_seg in segments[1:]:
#         last_word_merged = merged_text.split()[-1] if merged_text else ""
#         first_word_next = next_seg.split()[0] if next_seg else ""

#         if last_word_merged.lower() == first_word_next.lower():
#             merged_text += " " + " ".join(next_seg.split()[1:])
#         else:
#             merged_text += " " + next_seg
            
#     return merged_text.strip()


# @app.websocket("/ws/{client_id}")
# async def websocket_endpoint(websocket: WebSocket, client_id: str):
#     """WebSocket endpoint for real-time audio streaming and transcription."""
#     await manager.connect(websocket, client_id)

#     interim_transcription_task: asyncio.Task = None
    
#     try:
#         while True:
#             message = await websocket.receive_text()
#             data = json.loads(message)

#             if data["type"] == "ping":
#                 await manager.send_personal_message(
#                     json.dumps({"type": "pong", "timestamp": datetime.now().isoformat()}),
#                     client_id
#                 )
#                 continue

#             if data["type"] == "start_recording":
#                 language_code = data.get("language", "en")
#                 supported_languages = ["en", "ms", "my", "zh", "id", "ne", "ta", "bn"]
#                 if language_code not in supported_languages:
#                     logger.warning(f"Unsupported language code '{language_code}' for client {client_id}. Defaulting to English.")
#                     language_code = "en"
                
#                 client_languages[client_id] = language_code
#                 logger.info(f"[{client_id}] Recording started with language: {client_languages[client_id]}")

#                 manager.audio_buffers[client_id] = []
#                 manager.is_recording[client_id] = True

#                 if interim_transcription_task and not interim_transcription_task.done():
#                     interim_transcription_task.cancel()
#                     try:
#                         await interim_transcription_task
#                     except asyncio.CancelledError:
#                         pass

#                 interim_transcription_task = asyncio.create_task(
#                     process_audio_for_client(client_id, language=client_languages[client_id])
#                 )

#                 await manager.send_personal_message(
#                     json.dumps({"type": "status", "message": "Recording started.", "language": language_code}),
#                     client_id
#                 )
#                 continue

#             if data["type"] == "stop_recording":
#                 manager.is_recording[client_id] = False
#                 if interim_transcription_task and not interim_transcription_task.done():
#                     interim_transcription_task.cancel()
#                     try:
#                         await interim_transcription_task
#                     except asyncio.CancelledError:
#                         pass
#                     interim_transcription_task = None

#                 final_segments = manager.transcribed_segments.get(client_id, [])
#                 final_transcription = merge_transcriptions(final_segments)

#                 if not final_transcription.strip():
#                     final_transcription = "No speech detected."

#                 await manager.send_personal_message(
#                     json.dumps({"type": "transcription", "text": final_transcription, "is_final": True}),
#                     client_id
#                 )

#                 manager.audio_buffers[client_id] = []
#                 manager.transcribed_segments[client_id] = []
                
#                 await manager.send_personal_message(
#                     json.dumps({"type": "status", "message": "Recording stopped."}),
#                     client_id
#                 )
#                 continue

#             if data["type"] == "audio_chunk":
#                 if not manager.is_recording.get(client_id, False):
#                     continue

#                 try:
#                     audio_bytes = base64.b64decode(data["audio"])
#                     audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
#                     manager.audio_buffers[client_id].append(audio_np)
#                 except Exception as e:
#                     logger.exception(f"Error receiving or converting audio chunk for {client_id}:")
#                     await manager.send_personal_message(
#                         json.dumps({"type": "error", "message": f"Audio processing error: {str(e)}"}),
#                         client_id
#                     )

#     except WebSocketDisconnect:
#         logger.info(f"Client {client_id} disconnected via WebSocketDisconnect.")
#         manager.is_recording[client_id] = False
#         if interim_transcription_task and not interim_transcription_task.done():
#             interim_transcription_task.cancel()
#             try:
#                 await interim_transcription_task
#             except asyncio.CancelledError:
#                 pass
#         manager.disconnect(client_id)
#     except Exception as e:
#         logger.exception(f"Unhandled WebSocket error for {client_id}:")
#         manager.is_recording[client_id] = False
#         if interim_transcription_task and not interim_transcription_task.done():
#             interim_transcription_task.cancel()
#             try:
#                 await interim_transcription_task
#             except asyncio.CancelledError:
#                 pass
#         manager.disconnect(client_id)

# @app.get("/", response_class=HTMLResponse)
# async def get_root():
#     """Serves the main index.html page from the static directory."""
#     return HTMLResponse(content=open("static/index.html").read())

# @app.get("/health")
# async def health_check():
#     """Health check endpoint."""
#     return {
#         "status": "healthy",
#         "model_loaded": MODEL_LOADED,
#         "device": str(DEVICE) if DEVICE else "not set"
#     }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)