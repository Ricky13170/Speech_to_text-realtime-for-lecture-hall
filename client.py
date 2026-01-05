#!/usr/bin/env python3
"""
ASR Thesis - Client Launcher
"""

MODAL_ROOT_URL = "https://huynguyenwork14--asr-thesis-asr-web.modal.run"

def main():
    # Tự động thêm /app/ để vào thẳng giao diện
    APP_URL = f"{MODAL_ROOT_URL.rstrip('/')}/app/" 

    print("\n" + "="*60)
    print("  ASR THESIS - CLIENT LAUNCHER")
    print("="*60)
    print()
    print(f"{APP_URL}")
    print()
    

if __name__ == "__main__":
    main()
