let websocket;
let audioContext;
let mediaStreamSource;
let scriptProcessor;
let audioStream;
let isRecording = false;
const clientId = 'client-' + crypto.randomUUID();
const sampleRate = 16000; // Fixed sample rate for Whisper models (16kHz)

/**
 * Establishes a WebSocket connection to the backend.
 */
function connectWebSocket() {
    const wsUrl = `ws://127.0.0.1:8000/ws/${clientId}`;
    websocket = new WebSocket(wsUrl);
    
    websocket.onopen = () => {
        console.log('WebSocket connected to', wsUrl);
        updateStatus('Connected to server', 'ready');
    };
    
    websocket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log('Received message:', data);
        if (data.type === 'transcription') {
            if (data.is_final) {
                displayFinalTranscription(data.text);
            } else {
                displayInterimTranscription(data.text);
            }
        } else if (data.type === 'status') {
            updateStatus(data.message, 
                data.message.includes('Recording') ? 'recording' : 
                data.message.includes('Processing') ? 'processing' : 'ready');
        } else if (data.type === 'error') {
            console.error('Server Error:', data.message);
            updateStatus('Error: ' + data.message, 'error');
            if (isRecording) stopRecording();
        } else if (data.type === 'pong') {
            console.log('Pong received at', data.timestamp);
        }
    };
    
    websocket.onclose = () => {
        console.log('WebSocket closed');
        updateStatus('Disconnected from server', 'error');
        if (isRecording) handleRecordingStopped();
    };

    websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
        updateStatus('WebSocket Error', 'error');
        if (isRecording) stopRecording();
    };
}

/**
 * Starts the audio recording and streaming to the WebSocket.
 */
async function startRecording() {
    try {
        console.log('Starting recording...');
        // Clear previous transcriptions on new recording session
        document.getElementById('interimTranscription').textContent = '';
        document.getElementById('finalTranscription').textContent = 'Listening for commands...';
        
        // Ensure WebSocket is connected before starting audio capture
        if (!websocket || websocket.readyState !== WebSocket.OPEN) {
            console.log('WebSocket not open, attempting to connect...');
            connectWebSocket();
            await new Promise((resolve, reject) => {
                const timeout = setTimeout(() => reject(new Error('WebSocket connection timed out.')), 5000);
                websocket.onopen = () => { clearTimeout(timeout); resolve(); };
                websocket.onerror = (e) => { clearTimeout(timeout); reject(e); };
            });
        }
        
        // Request microphone access from the user
        audioStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                sampleRate: sampleRate,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            },
            video: false
        }).catch(err => {
            console.error('getUserMedia error:', err.name, err.message);
            throw new Error('Microphone access denied: ' + err.message);
        });

        // Create AudioContext for processing audio stream
        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: sampleRate });
        mediaStreamSource = audioContext.createMediaStreamSource(audioStream);
        
        // Create a ScriptProcessorNode for direct audio processing
        scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
        scriptProcessor.onaudioprocess = (event) => {
            if (!isRecording) return;
            console.log('Processing audio chunk, length:', event.inputBuffer.length);
            const audioData = event.inputBuffer.getChannelData(0);
            const audioInt16 = new Int16Array(audioData.length);
            for (let i = 0; i < audioData.length; i++) {
                audioInt16[i] = Math.min(32767, Math.max(-32768, audioData[i] * 32768));
            }
            if (websocket && websocket.readyState === WebSocket.OPEN) {
                websocket.send(JSON.stringify({
                    type: 'audio_chunk',
                    audio: arrayBufferToBase64(audioInt16.buffer)
                }));
                console.log('Sent audio chunk');
            }
        };
        
        // Connect nodes: source -> processor -> destination
        mediaStreamSource.connect(scriptProcessor);
        scriptProcessor.connect(audioContext.destination);

        // Inform the backend that recording has started
        if (websocket && websocket.readyState === WebSocket.OPEN) {
            websocket.send(JSON.stringify({
                type: 'start_recording',
                sample_rate: sampleRate
            }));
            console.log('Sent start_recording message');
        }
        
        isRecording = true;
        document.getElementById('startBtn').disabled = true;
        document.getElementById('stopBtn').disabled = false;
        updateStatus('Recording... Speak your command!', 'recording');
        
    } catch (error) {
        console.error('Error starting recording:', error);
        updateStatus('Error: Could not access microphone. Please check permissions.', 'error');
        handleRecordingStopped();
    }
}

/**
 * Stops the audio recording and sends a stop signal to the backend.
 */
function stopRecording() {
    if (!isRecording) return;
    
    isRecording = false;
    
    if (websocket && websocket.readyState === WebSocket.OPEN) {
        websocket.send(JSON.stringify({
            type: 'stop_recording'
        }));
        console.log('Sent stop_recording message');
    }
    
    handleRecordingStopped();
    updateStatus('Processing final command...', 'processing');
}

/**
 * Cleans up local audio recording resources (microphone, audio context).
 */
function handleRecordingStopped() {
    if (audioStream) {
        audioStream.getTracks().forEach(track => track.stop());
        audioStream = null;
    }
    if (scriptProcessor) {
        scriptProcessor.disconnect();
        scriptProcessor = null;
    }
    if (mediaStreamSource) {
        mediaStreamSource.disconnect();
        mediaStreamSource = null;
    }
    if (audioContext && audioContext.state !== 'closed') {
        audioContext.close();
        audioContext = null;
    }
    document.getElementById('startBtn').disabled = false;
    document.getElementById('stopBtn').disabled = true;
}

/**
 * Converts an ArrayBuffer to a Base64 string.
 * @param {ArrayBuffer} buffer The ArrayBuffer to convert.
 * @returns {string} The Base64 encoded string.
 */
function arrayBufferToBase64(buffer) {
    let binary = '';
    const bytes = new Uint8Array(buffer);
    const len = bytes.byteLength;
    for (let i = 0; i < len; i++) {
        binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
}

/**
 * Updates the status message displayed on the page.
 * @param {string} message The message to display.
 * @param {string} type A class name (e.g., 'ready', 'recording', 'error') for styling.
 */
function updateStatus(message, type) {
    const statusDiv = document.getElementById('status');
    statusDiv.textContent = message;
    statusDiv.className = 'status ' + type;
}

/**
 * Displays interim transcription results.
 * @param {string} text The interim transcription text.
 */
function displayInterimTranscription(text) {
    const interimDiv = document.getElementById('interimTranscription');
    interimDiv.textContent = text;
}

/**
 * Displays the final transcription result and clears interim.
 * @param {string} text The final transcription text.
 */
function displayFinalTranscription(text) {
    const finalDiv = document.getElementById('finalTranscription');
    finalDiv.textContent = text;
    document.getElementById('interimTranscription').textContent = '';
}

// Initialize UI and WebSocket connection when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', () => {
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    console.log('Buttons found:', startBtn, stopBtn);
    if (startBtn) {
        startBtn.addEventListener('click', () => {
            console.log('Start button clicked');
            startRecording();
        });
    }
    if (stopBtn) {
        stopBtn.addEventListener('click', stopRecording);
    }
    connectWebSocket();
});

// Clean up resources when the page is unloaded
window.addEventListener('beforeunload', () => {
    if (isRecording) stopRecording();
    if (websocket && websocket.readyState === WebSocket.OPEN) websocket.close();
});