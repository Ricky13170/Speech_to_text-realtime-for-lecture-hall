import os
from datetime import datetime
import time

class VTTLogger:
    def __init__(self, filename=None):
        if filename is None:
            filename = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.vtt"
        self.filename = filename
        self.start_time = time.time() # Use time.time() for easier diff calculation
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.filename) if os.path.dirname(self.filename) else ".", exist_ok=True)
        
        # Write WebVTT header
        with open(self.filename, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
        
        print(f"[VTTLogger] Logging to {self.filename}")

    def format_time(self, seconds):
        """Convert seconds (float) to VTT format (HH:MM:SS.mmm)"""
        # Handle negative or zero time gracefully
        if seconds < 0: seconds = 0
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"

    def log_segment(self, start_ts, end_ts, text):
        """
        Log a completed subtitle segment to the file.
        start_ts, end_ts: Absolute timestamps (time.time())
        """
        if not text or not text.strip():
            return
            
        # Calculate relative time from session start
        rel_start = start_ts - self.start_time
        rel_end = end_ts - self.start_time
        
        time_str = f"{self.format_time(rel_start)} --> {self.format_time(rel_end)}"
        
        try:
            with open(self.filename, "a", encoding="utf-8") as f:
                f.write(f"{time_str}\n{text}\n\n")
            print(f"[LOG] {time_str} : {text}")
        except Exception as e:
            print(f"[VTTLogger] Error writing to file: {e}")
