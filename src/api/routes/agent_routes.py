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
KILL_SWITCH_PATH = ROOT_DIR / ".kill_switch"
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
        pid=state.get("pid") if (running and state) else None,
        uptime_seconds=uptime,
        last_cycle=last_cycle,
        environment=env,
        llm_provider=settings.get("LLM_PROVIDER", "unknown"),
        llm_model=settings.get("LLM_MODEL", "unknown"),
        kill_switch_active=KILL_SWITCH_PATH.exists(),
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

    popen_kwargs = dict(
        cwd=str(ROOT_DIR),
        stdout=open(ROOT_DIR / "agent.log", "a"),
        stderr=subprocess.STDOUT,
        env=env_vars,
    )
    # On Windows, isolate the child from the parent's console so that
    # CTRL+C / console events don't propagate back and kill the API server.
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    proc = subprocess.Popen(cmd, **popen_kwargs)
    
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


@router.post("/decisions/clear")
async def clear_decisions(_: User = Depends(get_current_user)):
    try:
        DECISIONS_PATH.write_text("", encoding="utf-8")
    except Exception as exc:
        logger.exception("Failed to clear decisions log")
        raise HTTPException(status_code=500, detail=f"Failed to clear decisions: {exc}") from exc
    return {"message": "Recent LLM decisions cleared"}


@router.post("/diary/clear")
async def clear_diary(_: User = Depends(get_current_user)):
    try:
        DIARY_PATH.write_text("", encoding="utf-8")
    except Exception as exc:
        logger.exception("Failed to clear trade diary")
        raise HTTPException(status_code=500, detail=f"Failed to clear diary: {exc}") from exc
    return {"message": "Trade diary cleared"}


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


@router.post("/logs/clear")
async def clear_logs(_: User = Depends(get_current_user)):
    log_path = ROOT_DIR / "agent.log"
    try:
        log_path.write_text("", encoding="utf-8")
    except Exception as exc:
        logger.exception("Failed to clear agent logs")
        raise HTTPException(status_code=500, detail=f"Failed to clear logs: {exc}") from exc
    return {"message": "Agent logs cleared"}


async def _build_exchange(db: AsyncSession):
    """Build a DeltaExchangeAPI instance using credentials stored in the DB."""
    import sys
    sys.path.insert(0, str(ROOT_DIR))
    from src.trading.delta_api import DeltaExchangeAPI

    result = await db.execute(select(Setting))
    s = {row.key: row.value or "" for row in result.scalars().all()}

    base_url = s.get("DELTA_BASE_URL", "")
    is_testnet = "testnet" in base_url

    if is_testnet:
        api_key = s.get("DELTA_TESTNET_API_KEY") or s.get("DELTA_API_KEY", "")
        api_secret = s.get("DELTA_TESTNET_API_SECRET") or s.get("DELTA_API_SECRET", "")
    else:
        api_key = s.get("DELTA_PROD_API_KEY") or s.get("DELTA_API_KEY", "")
        api_secret = s.get("DELTA_PROD_API_SECRET") or s.get("DELTA_API_SECRET", "")

    try:
        leverage = int(s.get("DELTA_LEVERAGE", "5"))
    except (ValueError, TypeError):
        leverage = 5

    return DeltaExchangeAPI(
        base_url=base_url or None,
        api_key=api_key or None,
        api_secret=api_secret or None,
        leverage=leverage,
    ), s


@router.post("/kill-switch")
async def activate_kill_switch(
    close_positions: bool = True,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """Emergency halt: stop agent, cancel all open orders, optionally close all positions."""
    # 1. Write the flag file so the agent loop won't trade if it's mid-cycle
    KILL_SWITCH_PATH.write_text(json.dumps({
        "activated_at": datetime.now(timezone.utc).isoformat(),
        "close_positions": close_positions,
    }))

    # 2. Stop the agent process gracefully
    agent_was_running = _is_agent_running()
    if agent_was_running:
        state = _get_agent_state()
        if state and "pid" in state:
            pid = state["pid"]
            try:
                import psutil
                p = psutil.Process(pid)
                p.terminate()
                try:
                    p.wait(timeout=8)
                except psutil.TimeoutExpired:
                    p.kill()
            except ImportError:
                try:
                    import signal as _sig
                    os.kill(pid, _sig.SIGTERM)
                except OSError:
                    pass
            except Exception as exc:
                logger.warning("Kill switch: could not stop agent cleanly: %s", exc)
        try:
            STATE_PATH.unlink(missing_ok=True)
        except OSError:
            pass

    # 3. Connect to exchange and cancel orders / close positions
    cancelled: list[str] = []
    closed: list[str] = []
    errors: list[str] = []

    try:
        exchange, settings = await _build_exchange(db)
        assets_str = settings.get("ASSETS", "BTC ETH SOL")
        assets = [a.strip() for a in (assets_str.split(",") if "," in assets_str else assets_str.split()) if a.strip()]

        try:
            await exchange.init_products(assets)

            # Cancel all open orders
            try:
                open_orders = await exchange.get_open_orders()
                seen_coins: set[str] = set()
                for o in open_orders:
                    coin = o.get("coin") or o.get("asset") or ""
                    if coin and coin not in seen_coins:
                        seen_coins.add(coin)
                        try:
                            await exchange.cancel_all_orders(coin)
                            cancelled.append(coin)
                        except Exception as exc:
                            errors.append(f"cancel {coin}: {exc}")
            except Exception as exc:
                errors.append(f"fetch open orders: {exc}")

            # Close all open positions with market orders
            if close_positions:
                try:
                    state_data = await exchange.get_user_state()
                    for pos in state_data.get("positions", []):
                        coin = pos.get("coin")
                        raw_size = pos.get("szi") or pos.get("size") or 0
                        size = float(raw_size)
                        if not coin or abs(size) < 1e-9:
                            continue
                        is_long = size > 0
                        entry_px = float(pos.get("entryPx") or pos.get("entry_price") or 0)
                        alloc_usd = abs(size) * entry_px if entry_px else abs(size)
                        try:
                            if is_long:
                                await exchange.place_sell_order(coin, alloc_usd)
                            else:
                                await exchange.place_buy_order(coin, alloc_usd)
                            closed.append(coin)
                        except Exception as exc:
                            errors.append(f"close {coin}: {exc}")
                except Exception as exc:
                    errors.append(f"fetch positions: {exc}")
        finally:
            await exchange.close()
    except Exception as exc:
        errors.append(f"exchange init: {exc}")

    # 4. Write kill switch event to diary
    diary_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": "kill_switch_activated",
        "agent_was_running": agent_was_running,
        "cancelled_orders_for": cancelled,
        "closed_positions_for": closed,
        "errors": errors,
    }
    try:
        with open(DIARY_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(diary_entry) + "\n")
    except Exception:
        pass

    logger.warning(
        "KILL SWITCH activated — agent_stopped=%s cancelled=%s closed=%s errors=%s",
        agent_was_running, cancelled, closed, errors,
    )

    return {
        "message": "Kill switch activated",
        "agent_stopped": agent_was_running,
        "cancelled_orders_for": cancelled,
        "closed_positions_for": closed,
        "errors": errors,
    }


@router.post("/kill-switch/reset")
async def reset_kill_switch(_: User = Depends(get_current_user)):
    """Clear the kill switch flag so the agent can be started again."""
    if not KILL_SWITCH_PATH.exists():
        raise HTTPException(status_code=400, detail="Kill switch is not active")
    try:
        KILL_SWITCH_PATH.unlink()
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not clear kill switch: {exc}") from exc

    logger.info("Kill switch reset — agent may be restarted")
    return {"message": "Kill switch cleared. You may now restart the agent."}
