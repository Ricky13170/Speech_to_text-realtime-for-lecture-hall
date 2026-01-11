const WS_RECONNECT_MS = 2000;
const WS_MAX_RETRIES = 5;
const WS_TIMEOUT_MS = 30000;
const PING_INTERVAL_MS = 30000;

function getWebSocketURL() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${protocol}//${window.location.host}/ws/transcribe`;
}

class SocketManager {
    constructor() {
        this.ws = null;
        this.connected = false;
        this.reconnectAttempts = 0;
        this.reconnectTimer = null;
        this.pingInterval = null;
        this.onMessage = null;
        this.onConnected = null;
        this.onDisconnected = null;
        this.onError = null;
    }

    async connect() {
        return new Promise((resolve, reject) => {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                resolve();
                return;
            }

            const url = getWebSocketURL();
            this.ws = new WebSocket(url);

            const timeout = setTimeout(() => {
                if (this.ws && this.ws.readyState !== WebSocket.OPEN) {
                    this.ws.close();
                    reject(new Error('Connection timeout'));
                }
            }, WS_TIMEOUT_MS);

            this.ws.onopen = () => {
                clearTimeout(timeout);
                this.connected = true;
                this.reconnectAttempts = 0;
                this.startHeartbeat();
                if (this.onConnected) this.onConnected();
                resolve();
            };

            this.ws.onclose = (event) => {
                this.connected = false;
                this.stopHeartbeat();
                if (this.onDisconnected) this.onDisconnected();
                if (event.code !== 1000) this.scheduleReconnect();
            };

            this.ws.onerror = (err) => {
                if (this.onError) this.onError(err);
            };

            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (data.type === 'pong') return;
                    if (this.onMessage) this.onMessage(data);
                } catch (e) { }
            };
        });
    }

    scheduleReconnect() {
        if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
        if (this.reconnectAttempts >= WS_MAX_RETRIES) {
            if (this.onError) this.onError(new Error('Max retries reached'));
            return;
        }
        this.reconnectAttempts++;
        this.reconnectTimer = setTimeout(() => this.connect().catch(() => { }), WS_RECONNECT_MS);
    }

    startHeartbeat() {
        this.stopHeartbeat();
        this.pingInterval = setInterval(() => {
            if (this.isConnected()) this.send('ping');
        }, PING_INTERVAL_MS);
    }

    stopHeartbeat() {
        if (this.pingInterval) {
            clearInterval(this.pingInterval);
            this.pingInterval = null;
        }
    }

    send(type, data = {}) {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return false;
        try {
            this.ws.send(JSON.stringify({ type, ...data }));
            return true;
        } catch (e) {
            return false;
        }
    }

    close() {
        this.stopHeartbeat();
        if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
        if (this.ws) {
            this.ws.close(1000, 'User close');
            this.ws = null;
        }
        this.connected = false;
        this.reconnectAttempts = 0;
    }

    isConnected() {
        return this.connected && this.ws && this.ws.readyState === WebSocket.OPEN;
    }
}
export { SocketManager };