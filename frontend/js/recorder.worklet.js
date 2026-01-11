class RecorderWorklet extends AudioWorkletProcessor {
    constructor() {
        super();
        this.bufferSize = 4096;
        this.buffer = new Float32Array(this.bufferSize);
        this.bufferIndex = 0;

        // VAD Parameters
        this.vadThreshold = 0.01; // Easy RMS threshold
        this.vadHangover = 10;   // Frames to keep after speech stops
        this.speechFrames = 0;
    }

    calculateRMS(data) {
        let sum = 0;
        for (let i = 0; i < data.length; i++) {
            sum += data[i] * data[i];
        }
        return Math.sqrt(sum / data.length);
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        if (input && input.length > 0) {
            const channelData = input[0];

            // 1. Simple VAD Gate
            const rms = this.calculateRMS(channelData);

            if (rms > this.vadThreshold) {
                if (this.speechFrames === 0) {
                    console.log('[Worklet] VAD: Speech Start (RMS:', rms.toFixed(4), ')');
                }
                this.speechFrames = this.vadHangover;
            } else if (this.speechFrames > 0) {
                this.speechFrames--;
                if (this.speechFrames === 0) {
                    console.log('[Worklet] VAD: Speech End');
                }
            }

            // Only process if speech active
            if (this.speechFrames > 0) {
                this.port.postMessage({
                    command: 'process',
                    inputBuffer: channelData,
                    rms: rms
                });
            }
        }
        return true;
    }
}

registerProcessor('recorder-worklet', RecorderWorklet);
