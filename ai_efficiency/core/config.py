"""
Configuration and Credential Management for AI Providers.
Provides secure local storage, provider defaults, validation, and connection testing.
Never exposes API keys in frontend code, logs, or git.
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
SETTINGS_FILE = CONFIG_DIR / "ai_settings.json"

PROVIDER_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "freellmapi": {
        "name": "FreeLLMAPI (Unified Local Router)",
        "base_url": "http://localhost:3001/v1",
        "models": [
            "gemini-3.1-flash-lite",
            "gemini-3.5-flash-lite",
            "gemini-flash-lite",
            "gemini-3.8-flash",
            "auto",
        ],
        "default_model": "gemini-3.1-flash-lite",
        "requires_key": True,
        "env_key": "FREELLMAPI_KEY",
    },
    "openai": {
        "name": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
        "default_model": "gpt-4o-mini",
        "requires_key": True,
        "env_key": "OPENAI_API_KEY",
    },
    "groq": {
        "name": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
        "default_model": "llama-3.3-70b-versatile",
        "requires_key": True,
        "env_key": "GROQ_API_KEY",
    },
    "ollama": {
        "name": "Ollama (Local)",
        "base_url": "http://localhost:11434/v1",
        "models": ["llama3.2", "llama3", "mistral", "qwen2.5-coder", "phi3"],
        "default_model": "llama3.2",
        "requires_key": False,
        "env_key": None,
    },
    "openrouter": {
        "name": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "models": [
            "openai/gpt-4o-mini",
            "anthropic/claude-3.5-sonnet",
            "meta-llama/llama-3.3-70b-instruct"
        ],
        "default_model": "openai/gpt-4o-mini",
        "requires_key": True,
        "env_key": "OPENROUTER_API_KEY",
    },
    "gemini": {
        "name": "Google Gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "models": [
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-2.0-flash-lite-preview-02-05",
        ],
        "default_model": "gemini-2.0-flash",
        "requires_key": True,
        "env_key": "GEMINI_API_KEY",
    },
    "deepseek": {
        "name": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "default_model": "deepseek-chat",
        "requires_key": True,
        "env_key": "DEEPSEEK_API_KEY",
    },
    "huggingface": {
        "name": "Hugging Face",
        "base_url": "https://router.huggingface.co/v1",
        "models": [
            "Qwen/Qwen2.5-Coder-32B-Instruct",
            "Qwen/Qwen2.5-72B-Instruct",
            "meta-llama/Llama-3.3-70B-Instruct",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            "mistralai/Mistral-7B-Instruct-v0.3",
        ],
        "default_model": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "requires_key": True,
        "env_key": "HF_TOKEN",
    },
    "custom": {
        "name": "Custom / Any Other Provider",
        "base_url": "",
        "models": [],
        "default_model": "",
        "requires_key": False,
        "env_key": None,
    },
}


@dataclass
class AISettings:
    provider: str = "openai"
    api_key: str = ""
    model_name: str = "gpt-4o-mini"
    base_url: str = ""

    def __post_init__(self):
        if not self.base_url and self.provider in PROVIDER_DEFAULTS:
            self.base_url = PROVIDER_DEFAULTS[self.provider]["base_url"]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AISettings":
        prov = data.get("provider", "openai")
        default_base = PROVIDER_DEFAULTS.get(prov, {}).get("base_url", "https://api.openai.com/v1")
        return cls(
            provider=prov,
            api_key=data.get("api_key", ""),
            model_name=data.get("model_name", "gpt-4o-mini"),
            base_url=data.get("base_url") or default_base,
        )


def load_settings(filepath: Optional[Path] = None) -> AISettings:
    """Loads settings from config/ai_settings.json, falling back to environment variables."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    target_file = Path(filepath) if filepath else SETTINGS_FILE
    if target_file.exists():
        try:
            with open(target_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                settings = AISettings.from_dict(data)
                # If API key is empty in file, check env var
                if not settings.api_key:
                    prov = settings.provider
                    env_var = PROVIDER_DEFAULTS.get(prov, {}).get("env_key")
                    if env_var and os.environ.get(env_var):
                        settings.api_key = os.environ.get(env_var, "")
                return settings
        except Exception:
            pass

    # Default settings checking env
    provider = "openai"
    env_key = os.environ.get("OPENAI_API_KEY", "")
    if not env_key and os.environ.get("GROQ_API_KEY"):
        provider = "groq"
        env_key = os.environ.get("GROQ_API_KEY", "")

    prov_meta = PROVIDER_DEFAULTS.get(provider, PROVIDER_DEFAULTS["openai"])
    return AISettings(
        provider=provider,
        api_key=env_key,
        model_name=prov_meta["default_model"],
        base_url=prov_meta["base_url"],
    )


def save_settings(settings: AISettings, filepath: Optional[Path] = None) -> None:
    """Saves AI settings securely to config/ai_settings.json."""
    target_file = Path(filepath) if filepath else SETTINGS_FILE
    target_file.parent.mkdir(parents=True, exist_ok=True)
    with open(target_file, "w", encoding="utf-8") as f:
        json.dump(settings.to_dict(), f, indent=2)


def sanitize_model_name(name: str) -> str:
    """Sanitizes model name by removing accidental UI copy-paste suffixes and surrounding quotes."""
    if not name:
        return ""
    cleaned = name.strip().strip("'\"`")
    # Common accidental copy artifacts from web UI copy buttons (e.g. HuggingFace 'Copy' button)
    suffixes = [" Copy", " copy", " (Copy)", " (copy)", " Copied", " copied"]
    for suffix in suffixes:
        if cleaned.endswith(suffix) and len(cleaned) > len(suffix):
            cleaned = cleaned[:-len(suffix)].strip()
    return cleaned


def validate_configuration(settings: AISettings) -> Tuple[bool, str]:
    """
    Validates whether the provider, API key, and model name are configured.
    Returns (is_valid, error_message).
    """
    if not settings.provider or settings.provider not in PROVIDER_DEFAULTS:
        return False, "Please select an AI provider."

    prov_meta = PROVIDER_DEFAULTS[settings.provider]
    clean_model = sanitize_model_name(settings.model_name)

    # Validate API key if required
    if prov_meta["requires_key"]:
        if not settings.api_key or not settings.api_key.strip():
            return False, f"API Key is missing for {prov_meta['name']}. Please enter your API key."

    # Validate Model name
    if not clean_model:
        return False, "Model name is missing. Please select or enter a model name."

    # Validate Base URL for custom
    if settings.provider == "custom" and (not settings.base_url or not settings.base_url.strip()):
        return False, "Base URL is required when using a Custom OpenAI-compatible provider."

    return True, "Configuration is complete and valid."


def test_connection(settings: AISettings) -> Tuple[bool, str]:
    """
    Tests live connection to the configured AI provider and model.
    Returns (success: bool, message: str).
    """
    is_valid, msg = validate_configuration(settings)
    if not is_valid:
        return False, msg

    prov_meta = PROVIDER_DEFAULTS.get(settings.provider, PROVIDER_DEFAULTS["custom"])
    base_url = settings.base_url.strip() if settings.base_url.strip() else prov_meta["base_url"]
    api_key = settings.api_key.strip() if settings.api_key.strip() else "ollama"
    clean_model = sanitize_model_name(settings.model_name)

    try:
        import openai
        client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url if base_url else None,
            timeout=10.0,
            max_retries=0,
        )

        # Execute minimal completion to verify credentials and model availability
        response = client.chat.completions.create(
            model=clean_model,
            messages=[{"role": "user", "content": "Reply with pong"}],
            max_tokens=50,
        )

        if response and response.choices:
            return True, f"Connection successful! Successfully verified model '{clean_model}' on {prov_meta['name']}."
        return False, "Connection succeeded but received an empty response."

    except ImportError:
        return False, "The 'openai' Python package is not installed."
    except Exception as e:
        err_type = type(e).__name__
        err_str = str(e)
        server_msg = ""
        if hasattr(e, "body") and isinstance(e.body, dict):
            server_msg = e.body.get("error", "") or e.body.get("message", "")
        elif hasattr(e, "message"):
            server_msg = str(e.message)

        if "402" in err_str or getattr(e, "status_code", None) == 402:
            detail = server_msg if server_msg else "Account requires paid credits or PRO subscription to use this model."
            return False, f"Payment Required (402): {detail} Please add credits or switch to a free provider like Groq or Google Gemini."
        if "AuthenticationError" in err_type or "401" in err_str or getattr(e, "status_code", None) == 401:
            return False, f"Authentication failed (401): Invalid API key for {prov_meta['name']}."
        if "NotFoundError" in err_type or "404" in err_str or getattr(e, "status_code", None) == 404:
            return False, f"Model not found (404): '{clean_model}' is unavailable on {prov_meta['name']}."
        if "RateLimitError" in err_type or "429" in err_str or getattr(e, "status_code", None) == 429:
            return False, f"Rate limit or quota exceeded (429) for this {prov_meta['name']} account."
        if "APIConnectionError" in err_type or "ConnectionRefused" in err_str:
            return False, f"Connection refused: Could not connect to endpoint '{base_url}'."

        msg = f"{detail}" if (server_msg and (detail := server_msg)) else err_str
        return False, f"Provider error ({err_type}): {msg}"
