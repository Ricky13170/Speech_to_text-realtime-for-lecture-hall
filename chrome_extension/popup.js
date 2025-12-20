// popup.js
let audioContext = null;
let mediaStream = null;
let socket = null;
let isRecording = false;

// Kết nối WebSocket
function connectWebSocket() {
    return new Promise((resolve, reject) => {
        socket = new WebSocket('ws://localhost:8765');
        socket.onopen = () => resolve();
        socket.onerror = (error) => reject(error);
    });
}

// Hàm capture audio (chạy TRONG POPUP)
async function startCaptureInPopup(tabId) {
    try {
        console.log('🎬 Popup - Starting capture for tab:', tabId);
        
        // CAPTURE TRỰC TIẾP TỪ POPUP
        const stream = await new Promise((resolve, reject) => {
            chrome.tabCapture.capture(
                {
                    audio: true,
                    video: false,
                    audioConstraints: {
                        mandatory: {
                            channelCount: 1,
                            sampleRate: 16000
                        }
                    }
                },
                (stream) => {
                    if (chrome.runtime.lastError) {
                        reject(new Error(chrome.runtime.lastError.message));
                    } else {
                        resolve(stream);
                    }
                }
            );
        });
        
        console.log('🎬 Popup - Stream obtained');
        
        // Xử lý audio
        audioContext = new AudioContext({ sampleRate: 16000 });
        const source = audioContext.createMediaStreamSource(stream);
        const processor = audioContext.createScriptProcessor(4096, 1, 1);
        
        processor.onaudioprocess = (e) => {
            if (!socket || socket.readyState !== WebSocket.OPEN) return;
            
            const inputData = e.inputBuffer.getChannelData(0);
            const int16Array = new Int16Array(inputData.length);
            
            for (let i = 0; i < inputData.length; i++) {
                let s = Math.max(-1, Math.min(1, inputData[i]));
                int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
            }
            
            socket.send(int16Array.buffer);
        };
        
        source.connect(processor);
        processor.connect(audioContext.destination);
        mediaStream = stream;
        isRecording = true;
        
        return true;
        
    } catch (error) {
        console.error('🎬 Popup - Capture error:', error);
        return false;
    }
}

// Sự kiện click
document.getElementById('startBtn').addEventListener('click', async () => {
    try {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        
        if (!tab) {
            alert('Không tìm thấy tab!');
            return;
        }
        
        console.log('🟢 Popup - Starting capture process...');
        
        // 1. Kết nối WebSocket
        await connectWebSocket();
        console.log('🟢 Popup - WebSocket connected');
        
        // 2. Capture audio TRỰC TIẾP trong popup
        const success = await startCaptureInPopup(tab.id);
        
        if (success) {
            updateUI(true);
            console.log('🟢 Popup - Capture started successfully');
            
            // Thông báo cho background biết trạng thái
            chrome.runtime.sendMessage({
                action: 'capture_started',
                tabId: tab.id
            });
        } else {
            alert('Không thể bắt đầu thu âm. Kiểm tra xem tab có âm thanh không?');
        }
        
    } catch (error) {
        console.error('Popup error:', error);
        alert('Lỗi: ' + error.message);
    }
});

document.getElementById('stopBtn').addEventListener('click', () => {
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
    }
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }
    if (socket) {
        socket.close();
        socket = null;
    }
    isRecording = false;
    updateUI(false);
    
    chrome.runtime.sendMessage({ action: 'capture_stopped' });
});

function updateUI(recording) {
    document.getElementById('status').textContent = 
        recording ? '🎤 Đang thu âm...' : '⏸️ Sẵn sàng';
    document.getElementById('startBtn').disabled = recording;
    document.getElementById('stopBtn').disabled = !recording;
}
