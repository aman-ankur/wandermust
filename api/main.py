"""FastAPI application factory for Discovery v2."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router


def create_app() -> FastAPI:
    app = FastAPI(title="Wandermust Discovery v2", version="2.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app
