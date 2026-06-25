#!/usr/bin/env python3
"""
Run the BasketIQ FastAPI server.

Usage:
    python run_api.py
"""

import subprocess
import sys

if __name__ == "__main__":
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "api.main:app",
        "--reload",
        "--port",
        "8000",
    ]
    subprocess.run(cmd, check=False)
