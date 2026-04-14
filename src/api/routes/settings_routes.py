"""Settings routes: CRUD for all configuration settings + env file sync."""

import os
import pathlib
import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.api.database import get_db
from src.api.models import Setting, User
from src.api.auth import get_current_user
from src.api.schemas import SettingOut, BulkSettingsUpdate, SettingUpdate, SwitchEnvRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/settings", tags=["settings"])

TESTNET_URL = "https://cdn-ind.testnet.deltaex.org"
MAINNET_URL = "https://api.india.delta.exchange"

# ─── Env File Write Removed as per user request ────────────────────────


# ─── Routes ──────────────────────────────────────────────────────────────────

@router.get("", response_model=List[SettingOut])
async def list_settings(
    category: str = None,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    query = select(Setting)
    if category:
        query = query.where(Setting.category == category)
    result = await db.execute(query.order_by(Setting.category, Setting.id))
    return result.scalars().all()


@router.get("/{key}", response_model=SettingOut)
async def get_setting(
    key: str,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    result = await db.execute(select(Setting).where(Setting.key == key))
    setting = result.scalar_one_or_none()
    if not setting:
        raise HTTPException(status_code=404, detail=f"Setting '{key}' not found")
    return setting


@router.put("", response_model=List[SettingOut])
async def bulk_update_settings(
    payload: BulkSettingsUpdate,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    updated = []
    for item in payload.settings:
        result = await db.execute(select(Setting).where(Setting.key == item.key))
        setting = result.scalar_one_or_none()
        if setting:
            setting.value = item.value or ""
            db.add(setting)
            updated.append(setting)

    await db.commit()
    return updated


@router.put("/{key}", response_model=SettingOut)
async def update_setting(
    key: str,
    payload: SettingUpdate,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    result = await db.execute(select(Setting).where(Setting.key == key))
    setting = result.scalar_one_or_none()
    if not setting:
        raise HTTPException(status_code=404, detail=f"Setting '{key}' not found")
    setting.value = payload.value or ""
    db.add(setting)
    await db.commit()
    await db.refresh(setting)
    return setting


@router.post("/actions/switch-env")
async def switch_environment(
    payload: SwitchEnvRequest,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    url = TESTNET_URL if payload.environment == "testnet" else MAINNET_URL
    result = await db.execute(select(Setting).where(Setting.key == "DELTA_BASE_URL"))
    setting = result.scalar_one_or_none()
    if not setting:
        raise HTTPException(status_code=404, detail="DELTA_BASE_URL setting not found")

    setting.value = url
    db.add(setting)
    await db.commit()

    return {"environment": payload.environment, "url": url}


@router.get("/actions/current-env")
async def get_current_env(
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    result = await db.execute(select(Setting).where(Setting.key == "DELTA_BASE_URL"))
    setting = result.scalar_one_or_none()
    if not setting:
        return {"environment": "testnet", "url": TESTNET_URL}
    url = setting.value or TESTNET_URL
    env = "mainnet" if "api.india" in url else "testnet"
    return {"environment": env, "url": url}
