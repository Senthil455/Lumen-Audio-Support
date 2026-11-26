"""All configuration comes from the environment. No secrets in the repo.

A .env file is loaded if present, but the process never requires one: with no
LLM key the service runs in offline mode using the deterministic extractive
fallback, which is what makes the demo runnable without a signup.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]

load_dotenv(ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _num(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:  # fail loudly at startup, not per request
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    app_name: str = "Lumen Audio Support Copilot"
    version: str = "1.0.0"

    # LLM provider (any OpenAI-compatible chat completions endpoint)
    llm_api_key: str = ""
    llm_base_url: str = "https://api.z.ai/api/paas/v4"
    llm_model: str = "glm-4-flash"
    llm_timeout_s: float = 20.0
    require_llm_key: bool = False

    # Cost accounting. Defaults match a cheap hosted model; 0 keeps the demo free.
    price_input_per_1k: float = 0.0001
    price_output_per_1k: float = 0.0001
    daily_budget_usd: float = 0.50

    # Serving behaviour
    rate_limit: str = "5/minute"
    agent_rate_limit: str = "3/minute"
    trust_proxy_headers: bool = False
    top_k: int = 4
    max_agent_steps: int = 4

    # Privacy: caller identifiers are hashed with this salt before logging.
    log_salt: str = field(default_factory=lambda: _default_salt())

    # MCP: the docs tool is delivered by a stdio MCP server by default.
    mcp_enabled: bool = True
    mcp_command: str = ""

    log_dir: Path = ROOT / "logs"
    runtime_dir: Path = ROOT / "data" / "runtime"
    cors_origins: tuple[str, ...] = ("*",)

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_api_key)

    @property
    def mode(self) -> str:
        return "llm" if self.llm_enabled else "offline-extractive"


def _default_salt() -> str:
    # A per-process random salt is fine for local runs; set LOG_SALT in
    # production so caller hashes stay stable across restarts.
    import secrets

    return os.environ.get("LOG_SALT") or secrets.token_hex(16)


def load_settings() -> Settings:
    settings = Settings(
        llm_api_key=_env("LLM_API_KEY"),
        llm_base_url=_env("LLM_BASE_URL", "https://api.z.ai/api/paas/v4"),
        llm_model=_env("LLM_MODEL", "glm-4-flash"),
        llm_timeout_s=_num("LLM_TIMEOUT_S", 20.0),
        require_llm_key=_flag("REQUIRE_LLM_KEY"),
        price_input_per_1k=_num("PRICE_INPUT_PER_1K", 0.0001),
        price_output_per_1k=_num("PRICE_OUTPUT_PER_1K", 0.0001),
        daily_budget_usd=_num("DAILY_BUDGET_USD", 0.50),
        rate_limit=_env("RATE_LIMIT", "5/minute"),
        agent_rate_limit=_env("AGENT_RATE_LIMIT", "3/minute"),
        trust_proxy_headers=_flag("TRUST_PROXY_HEADERS"),
        top_k=int(_num("TOP_K", 4)),
        max_agent_steps=int(_num("MAX_AGENT_STEPS", 4)),
        mcp_enabled=_flag("MCP_ENABLED", True),
        mcp_command=_env("MCP_COMMAND"),
        log_dir=Path(_env("LOG_DIR") or ROOT / "logs"),
        runtime_dir=Path(_env("RUNTIME_DIR") or ROOT / "data" / "runtime"),
    )
    if settings.require_llm_key and not settings.llm_api_key:
        # The explicit production check from the lesson: fail at boot, not on
        # the first customer request.
        raise RuntimeError("LLM_API_KEY is required (REQUIRE_LLM_KEY=true) but missing")
    return settings


settings = load_settings()
