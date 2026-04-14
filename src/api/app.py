"""FastAPI application entrypoint."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pathlib

from src.api.database import init_db, AsyncSessionLocal
from src.api.seed import run_seed
from src.api.routes.auth_routes import router as auth_router
from src.api.routes.settings_routes import router as settings_router
from src.api.routes.agent_routes import router as agent_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FRONTEND_DIST = pathlib.Path(__file__).parent.parent.parent / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Trading Agent API…")
    await init_db()
    async with AsyncSessionLocal() as db:
        await run_seed(db)
    logger.info("Database ready.")
    yield
    logger.info("API shutting down.")


app = FastAPI(
    title="Trading Agent Dashboard API",
    description="REST API for the Delta Exchange LLM Trading Agent dashboard",
    version="1.0.0",
    lifespan=lifespan,
)

# ─── CORS ─────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server
        "http://localhost:3001",   # alternative dev port
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── API Routers ──────────────────────────────────────────────────────────────

app.include_router(auth_router)
app.include_router(settings_router)
app.include_router(agent_router)


# ─── Health check ─────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "trading-agent-api"}


# ─── Serve built React frontend (production) ──────────────────────────────────

if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        index = FRONTEND_DIST / "index.html"
        return FileResponse(str(index))
