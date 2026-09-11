from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router

app = FastAPI(
    title="cg-gateway: ChatGPT OpenAI-Compatible API Proxy",
    description="Production-grade reverse gateway for chatgpt.com supporting OpenAI API SDK protocol, multi-turn Smart Session Pool, and Sentinel PoW bypass.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
