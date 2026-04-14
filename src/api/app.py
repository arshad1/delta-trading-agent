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
import subprocess
import shutil
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FRONTEND_DIST = pathlib.Path(__file__).parent.parent.parent / "frontend" / "dist"


def build_frontend():
    """Run npm install & npm run build in the frontend directory."""
    frontend_dir = pathlib.Path(__file__).parent.parent.parent / "frontend"
    logger.info("Building frontend with npm...")
    
    # Needs shell=True on Windows to find npm.cmd correctly in subprocess
    use_shell = os.name == "nt"
    npm_cmd = "npm" if use_shell else shutil.which("npm")
    
    if not npm_cmd and not use_shell:
        logger.warning("npm not found. Skipping frontend build.")
        return
        
    try:
        # Check if node_modules exists, otherwise install dependencies
        if not (frontend_dir / "node_modules").exists():
            logger.info("Running npm install...")
            subprocess.run([npm_cmd, "install"], cwd=str(frontend_dir), check=True, shell=use_shell)
        
        logger.info("Running npm run build...")
        subprocess.run([npm_cmd, "run", "build"], cwd=str(frontend_dir), check=True, shell=use_shell)
        logger.info("Frontend build complete.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Frontend build failed with exit code {e.returncode}.")
    except Exception as e:
        logger.error(f"Could not build frontend: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Trading Agent API…")
    
    # 1. Automatically build frontend on startup
    build_frontend()

    # 2. Setup database
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

@app.get("/{full_path:path}", include_in_schema=False)
async def serve_spa(full_path: str):
    """Catch-all route that serves the React SPA index.html for all unknown paths."""
    
    # If a direct file is requested (like /assets/main.js or /favicon.ico), serve it
    if full_path:
        requested_file = FRONTEND_DIST / full_path
        # Prevent path traversal attacks
        try:
            if FRONTEND_DIST.resolve() in requested_file.resolve().parents or FRONTEND_DIST.resolve() == requested_file.resolve():
                if requested_file.is_file():
                    return FileResponse(str(requested_file))
                # Explicitly return 404 for missing static assets to prevent sending index.html
                if full_path.startswith("assets/"):
                    from fastapi import HTTPException
                    raise HTTPException(status_code=404, detail="Asset not found")
        except Exception:
            pass

    # If the file doesn't exist, fallback to index.html (React Router will handle the 404)
    index = FRONTEND_DIST / "index.html"
    if not index.exists():
        return {
            "error": "Frontend not built yet.",
            "message": "The API server tried to build the frontend, but dist/index.html was not found. Please run 'npm install' and 'npm run build' inside the frontend directory manually."
        }
    return FileResponse(str(index))
