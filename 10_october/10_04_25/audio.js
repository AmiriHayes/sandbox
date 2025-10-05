const recordButton = document.getElementById('recordButton');
const textbox = document.getElementById('textbox');
const status = document.getElementById('status');

let mediaRecorder;
let audioChunks = [];
let isRecording = false;

const API_URL = 'https://amirihayes-transcription-app.static.hf.space'; 

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true});
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        }

        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/webm'});
            await sendAudioToBackend(audioBlob);
            stream.getTracks().forEach(track => track.stop());
        }

        mediaRecorder.start();
        isRecording = true;
        recordButton.classList.add('recording');
        recordButton.textContent = 'Recording...';
        status.textContent = 'Status: Recording...';
    } catch (error) {
        console.error('Error starting recording:', error);
        status.textContent = "Error: Could not access microphone.";
    }
}
function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;
        recordButton.classList.remove('recording');
        recordButton.textContent = 'Hold to Record';
        status.textContent = 'Processing...';
    }
}
async function sendAudioToBackend(audioBlob) {
    try {
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.webm');

        const response = await fetch(API_URL, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) { throw new Error(`HTTP error! status: ${response.status}`); }
        const data = await response.json();

        if (data.text) {
            textbox.value = textbox.value + (textbox.value ? ' ' : '') + data.text;
            status.textContent = 'Transcription complete!';
        } else if (data.error) {
            status.textContent = `Error: ${data.error}`;
        }

        setTimeout(() => {
            status.textContent = '';
        }, 3_000);
    } catch (error) {
        console.error('Error sending audio to backend:', error);
        status.textContent = 'Error: Could not process audio.';
    }
}
recordButton.addEventListener('mousedown', startRecording);
recordButton.addEventListener('mouseup', stopRecording);
recordButton.addEventListener('mouseleave', stopRecording);
recordButton.addEventListener('touchstart', (e) => {
    e.preventDefault();
    startRecording();
});
recordButton.addEventListener('touchend', (e) => {
    e.preventDefault();
    stopRecording();
});