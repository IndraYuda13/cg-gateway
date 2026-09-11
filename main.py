#!/usr/bin/env python3
"""
Main entry point for cg-gateway.
"""
from server import main

if __name__ == "__main__":
    import uvicorn
    from app.main import app
    from app.config import HOST, PORT
    print(f"Starting cg-gateway on {HOST}:{PORT}...")
    uvicorn.run(app, host=HOST, port=PORT)
