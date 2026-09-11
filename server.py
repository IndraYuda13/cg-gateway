#!/usr/bin/env python3
"""
Launcher for cg-gateway service.
Runs Uvicorn on configured HOST and PORT (default 8560).
"""
import uvicorn
from app.main import app
from app.config import HOST, PORT

if __name__ == "__main__":
    print(f"Starting cg-gateway on {HOST}:{PORT}...")
    uvicorn.run(app, host=HOST, port=PORT)
