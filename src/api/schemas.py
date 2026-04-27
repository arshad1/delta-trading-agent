"""Pydantic schemas for request/response validation."""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, field_validator


# ─── Auth ────────────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str


class UserOut(BaseModel):
    id: int
    username: str
    email: Optional[str]
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


# ─── Settings ────────────────────────────────────────────────────────────────

class SettingOut(BaseModel):
    id: int
    key: str
    value: Optional[str]
    category: str
    label: str
    description: str
    is_secret: bool
    updated_at: datetime

    model_config = {"from_attributes": True}


class SettingUpdate(BaseModel):
    key: str
    value: Optional[str] = ""


class BulkSettingsUpdate(BaseModel):
    settings: List[SettingUpdate]


class SwitchEnvRequest(BaseModel):
    environment: str  # "testnet" or "mainnet"

    @field_validator("environment")
    @classmethod
    def validate_env(cls, v: str) -> str:
        if v not in ("testnet", "mainnet"):
            raise ValueError("environment must be 'testnet' or 'mainnet'")
        return v


# ─── Agent ────────────────────────────────────────────────────────────────────

class AgentStatusOut(BaseModel):
    running: bool
    pid: Optional[int]
    uptime_seconds: Optional[float]
    last_cycle: Optional[str]
    environment: str
    llm_provider: str
    llm_model: str
    kill_switch_active: bool = False
