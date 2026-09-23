"""talos.config.json and .talos/secrets.json (the LLM key and the C3 API key)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path

ENV_KEYS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY",
            "google": "GEMINI_API_KEY", "openrouter": "OPENROUTER_API_KEY",
            "custom": "TALOS_CUSTOM_API_KEY"}


class ConfigError(RuntimeError):
    pass


@dataclass
class Config:
    provider: str
    model: str
    mode: str
    api_base: str | None
    backend: str = "modal"
    local_cpus: int | None = None       # local backend: the job container's --cpus
    local_memory_gib: int | None = None  # local backend: the job container's --memory, GiB
    config_path: Path | None = None
    secrets_path: Path | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("config_path")
        d.pop("secrets_path")
        # Written only when set, so a Modal or C3 config says nothing about a local container.
        return {k: v for k, v in d.items() if not (k.startswith("local_") and v is None)}


def load(root: Path) -> Config:
    root = Path(root)
    cp = root / "talos.config.json"
    if not cp.exists():
        raise ConfigError("talos.config.json not found; run `talos setup` first")
    d = json.loads(cp.read_text(encoding="utf-8"))
    return Config(provider=d["provider"], model=d["model"], mode=d.get("mode", "single-shot"),
                  api_base=d.get("api_base"), backend=d.get("backend", "modal"),
                  local_cpus=d.get("local_cpus"), local_memory_gib=d.get("local_memory_gib"),
                  config_path=cp,
                  secrets_path=root / ".talos" / "secrets.json")


def save(root: Path, config: Config, api_key: str | None, c3_api_key: str | None = None) -> None:
    root = Path(root)
    (root / "talos.config.json").write_text(json.dumps(config.to_dict(), indent=1) + "\n",
                                            encoding="utf-8", newline="\n")
    secrets = {k: v for k, v in (("api_key", api_key), ("c3_api_key", c3_api_key)) if v}
    sp = root / ".talos" / "secrets.json"
    if not secrets:
        # A key left out of this setup is dropped, not kept: a stale c3_api_key would otherwise
        # override the login session that setup just checked.
        sp.unlink(missing_ok=True)
        return
    sp.parent.mkdir(exist_ok=True)
    # Create the file 0600 rather than writing at the process umask and narrowing it after:
    # between write_text and chmod the key was readable by anyone on the machine.
    fd = os.open(sp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(secrets) + "\n")
    os.chmod(sp, 0o600)  # an existing file keeps its old mode through O_CREAT


def resolve_api_key(config: Config) -> str | None:
    if config.secrets_path and config.secrets_path.exists():
        key = json.loads(config.secrets_path.read_text(encoding="utf-8")).get("api_key")
        if key:
            return key
    env = ENV_KEYS.get(config.provider)
    val = os.environ.get(env, "") if env else ""
    return val or None


def resolve_c3_api_key(config: Config | None) -> str | None:
    """The C3 API key: .talos/secrets.json, then C3_API_KEY. None means the `c3` CLI uses its
    `c3 login` session. `config` is None for a `talos compile` run where no config exists."""
    if config and config.secrets_path and config.secrets_path.exists():
        key = json.loads(config.secrets_path.read_text(encoding="utf-8")).get("c3_api_key")
        if key:
            return key
    return os.environ.get("C3_API_KEY") or None
