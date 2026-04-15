"""Agent control and data relay routes."""

import os
import json
import sys
import signal
import asyncio
import logging
import pathlib
import subprocess
import time
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.api.database import get_db
from src.api.models import User, Setting
from src.api.auth import get_current_user
from src.api.schemas import AgentStatusOut

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/agent", tags=["agent"])

# ─── In-memory agent process state ───────────────────────────────────────────

# We use a state file instead of in-memory globals so that the agent state 
# persists across uvicorn reloads or multiple Gunicorn workers.
ROOT_DIR = pathlib.Path(__file__).parent.parent.parent.parent
STATE_PATH = ROOT_DIR / ".agent_state"
DIARY_PATH = ROOT_DIR / "diary.jsonl"
DECISIONS_PATH = ROOT_DIR / "decisions.jsonl"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _get_agent_state() -> Optional[dict]:
    if not STATE_PATH.exists():
        return None
    try:
        return json.loads(STATE_PATH.read_text())
    except Exception:
        return None


def _is_agent_running() -> bool:
    state = _get_agent_state()
    if not state or "pid" not in state:
        return False
        
    pid = state["pid"]
    try:
        import psutil
        return psutil.pid_exists(pid)
    except ImportError:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def _read_jsonl_tail(path: pathlib.Path, limit: int = 100) -> list:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    tail = lines[-limit:] if len(lines) > limit else lines
    result = []
    for line in tail:
        try:
            result.append(json.loads(line))
        except Exception:
            pass
    return result


# ─── Routes ──────────────────────────────────────────────────────────────────

@router.get("/status", response_model=AgentStatusOut)
async def get_status(
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    running = _is_agent_running()
    uptime = None
    state = _get_agent_state()
    
    if running and state and "started_at" in state:
        uptime = time.time() - state["started_at"]
    elif state and not running:
        # Cleanup stale state wrapper
        try:
            STATE_PATH.unlink()
        except OSError:
            pass

    last_cycle = None
    decisions = _read_jsonl_tail(DECISIONS_PATH, 1)
    if decisions:
        last_cycle = decisions[-1].get("timestamp")

    # Read current LLM provider and model from DB
    result = await db.execute(
        select(Setting).where(Setting.key.in_(["LLM_PROVIDER", "LLM_MODEL", "DELTA_BASE_URL"]))
    )
    settings = {s.key: s.value for s in result.scalars().all()}
    base_url = settings.get("DELTA_BASE_URL", "")
    env = "mainnet" if "api.india" in base_url else "testnet"

    return AgentStatusOut(
        running=running,
        pid=_agent_proc.pid if running else None,
        uptime_seconds=uptime,
        last_cycle=last_cycle,
        environment=env,
        llm_provider=settings.get("LLM_PROVIDER", "unknown"),
        llm_model=settings.get("LLM_MODEL", "unknown"),
    )


@router.post("/start")
async def start_agent(
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    if _is_agent_running():
        raise HTTPException(status_code=400, detail="Agent is already running")

    # Read ALL settings from DB to pass to agent
    result = await db.execute(select(Setting))
    all_settings = result.scalars().all()
    settings_dict = {s.key: s.value or "" for s in all_settings}

    assets_str = settings_dict.get("ASSETS", "BTC ETH SOL")
    assets = assets_str.split(",") if "," in assets_str else assets_str.split()
    interval = settings_dict.get("INTERVAL", "5m")

    cmd = [
        sys.executable, "-m", "src.main",
        "--assets", *assets,
        "--interval", interval,
    ]

    # Inject all DB settings as environment variables so config_loader picks them up
    env_vars = os.environ.copy()
    for k, v in settings_dict.items():
        if v:  # only export non-empty values
            env_vars[k] = str(v)

    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT_DIR),
        stdout=open(ROOT_DIR / "agent.log", "a"),
        stderr=subprocess.STDOUT,
        env=env_vars,
    )
    
    # Save the agent process PID and start time
    started_at = time.time()
    STATE_PATH.write_text(json.dumps({"pid": proc.pid, "started_at": started_at}))
    
    logger.info("Agent started with PID %d", proc.pid)
    return {"message": "Agent started", "pid": proc.pid}


@router.post("/stop")
async def stop_agent(_: User = Depends(get_current_user)):
    if not _is_agent_running():
        raise HTTPException(status_code=400, detail="Agent is not running")
        
    state = _get_agent_state()
    if not state or "pid" not in state:
        raise HTTPException(status_code=400, detail="Agent state is missing or corrupted")
        
    pid = state["pid"]
    try:
        try:
            import psutil
            p = psutil.Process(pid)
            p.terminate()
            p.wait(timeout=10)
        except ImportError:
            if sys.platform == "win32":
                os.kill(pid, signal.SIGTERM)
            else:
                os.kill(pid, signal.SIGTERM)
    except Exception as exc:
        logger.warning("Could not stop agent gracefully: %s", exc)
        # Fallback to extreme kill
        try:
            os.kill(pid, signal.SIGKILL if not sys.platform == "win32" else signal.SIGTERM)
        except OSError:
            pass
            
    try:
        if STATE_PATH.exists():
            STATE_PATH.unlink()
    except OSError:
        pass

    return {"message": "Agent stopped"}


@router.get("/diary")
async def get_diary(
    limit: int = 100,
    _: User = Depends(get_current_user),
):
    return {"entries": _read_jsonl_tail(DIARY_PATH, limit)}


@router.get("/decisions")
async def get_decisions(
    limit: int = 50,
    _: User = Depends(get_current_user),
):
    return {"entries": _read_jsonl_tail(DECISIONS_PATH, limit)}


@router.get("/logs")
async def get_logs(
    lines: int = 200,
    _: User = Depends(get_current_user),
):
    log_path = ROOT_DIR / "agent.log"
    if not log_path.exists():
        return {"content": ""}
    content = log_path.read_text(encoding="utf-8", errors="replace")
    tail_lines = content.strip().splitlines()[-lines:]
    return {"content": "\n".join(tail_lines)}
