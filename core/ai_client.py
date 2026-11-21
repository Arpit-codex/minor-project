"""
AI Client module.
Configurable client for free-tier AI APIs (Google Gemini) and a deterministic Mock provider.
Gracefully handles rate limits, network timeouts, authentication issues, and missing keys.
"""

import os
import json
import time
from typing import Dict, Any, Optional, List
from config import Config


class AIResponse:
    """Standardized AI response container."""

    def __init__(
        self,
        content: str,
        provider: str,
        model: str,
        is_error: bool = False,
        error_type: Optional[str] = None,
        raw: Any = None
    ):
        self.content = content
        self.provider = provider
        self.model = model
        self.is_error = is_error
        self.error_type = error_type
        self.raw = raw

    def __str__(self) -> str:
        return self.content


class BaseAIProvider:
    """Abstract base class for AI providers."""

    def generate(self, prompt: str, system_instruction: Optional[str] = None) -> AIResponse:
        raise NotImplementedError

    def is_available(self) -> bool:
        raise NotImplementedError


class MockAIProvider(BaseAIProvider):
    """Deterministic Mock AI provider for automated testing and offline development."""

    def __init__(self):
        self.provider_name = "mock"
        self.model_name = "mock-contextual-v1"
        self.simulate_rate_limit: bool = False
        self.simulate_network_error: bool = False

    def is_available(self) -> bool:
        return True

    def generate(self, prompt: str, system_instruction: Optional[str] = None) -> AIResponse:
        if self.simulate_rate_limit:
            return AIResponse(
                content="[Error: Free-tier API rate limit reached (HTTP 429: RESOURCE_EXHAUSTED). Please wait a moment.]",
                provider=self.provider_name,
                model=self.model_name,
                is_error=True,
                error_type="rate_limit"
            )

        if self.simulate_network_error:
            return AIResponse(
                content="[Error: Network connection failed. Please check your internet connection.]",
                provider=self.provider_name,
                model=self.model_name,
                is_error=True,
                error_type="network_error"
            )

        # Extract current user question if structured prompt
        p_lower = prompt.lower()
        q_lower = p_lower
        if "### current user question:" in p_lower:
            q_lower = p_lower.split("### current user question:")[-1].strip()

        # 1. Pronoun / follow-up resolution test: "make it run with less RAM"
        if "less ram" in q_lower and ("language model" in p_lower or "model" in p_lower):
            answer = "To run the language model with less RAM on your laptop, you can apply 4-bit or 8-bit quantization (such as GGUF/llama.cpp or bitsandbytes), offload layers to disk, or use a smaller parameter variant."

        # 2. Project memory test
        elif "project" in q_lower and ("architecture" in q_lower or "stack" in q_lower):
            answer = "Your current project is a Local Memory-Based Contextual AI Assistant built with Python, SQLite, and categorized Markdown storage."

        # 3. College memory test
        elif "college" in q_lower or "university" in q_lower or "degree" in q_lower:
            if "national institute of technology" in p_lower or "computer science" in p_lower:
                answer = "Based on your stored records, you are studying Computer Science at the National Institute of Technology."
            else:
                answer = "I do not have your college details stored in your local memory. Could you please share your university or college name?"

        # 4. Capital of Italy
        elif "capital of italy" in q_lower:
            answer = "The capital of Italy is Rome."

        # 5. Preference test
        elif "dark mode" in q_lower or "response style" in q_lower:
            answer = "Noted your preference! I will ensure my responses adhere to your requested style."

        # 6. Clarification test
        elif "clarify" in q_lower or "which option" in q_lower:
            answer = "Could you clarify which specific item you would like me to elaborate on?"

        # 7. Default general contextual response
        else:
            answer = "I understood your question regarding the topic. Here is the context-aware explanation tailored to your setup."

        return AIResponse(
            content=answer,
            provider=self.provider_name,
            model=self.model_name,
            is_error=False
        )


class GeminiAIProvider(BaseAIProvider):
    """Google Gemini AI Provider utilizing the modern google-genai SDK."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or Config.GEMINI_API_KEY
        self.model_name = model or Config.GEMINI_MODEL
        self.provider_name = "gemini"
        self._client = None

        if self.api_key:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception:
                self._client = None

    def is_available(self) -> bool:
        return bool(self._client and self.api_key)

    def generate(self, prompt: str, system_instruction: Optional[str] = None) -> AIResponse:
        if not self.is_available():
            return AIResponse(
                content="[Google Gemini API is not configured. Please add your GEMINI_API_KEY in the .env file or run in mock mode.]",
                provider=self.provider_name,
                model=self.model_name,
                is_error=True,
                error_type="missing_api_key"
            )

        try:
            from google.genai import types
            config_params = {}
            if system_instruction:
                config_params["system_instruction"] = system_instruction

            config = types.GenerateContentConfig(**config_params) if config_params else None

            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config
            )

            if response and response.text:
                return AIResponse(
                    content=response.text.strip(),
                    provider=self.provider_name,
                    model=self.model_name,
                    is_error=False,
                    raw=response
                )
            else:
                return AIResponse(
                    content="[Received empty response from Gemini model.]",
                    provider=self.provider_name,
                    model=self.model_name,
                    is_error=True,
                    error_type="empty_response"
                )

        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str:
                return AIResponse(
                    content=f"[Gemini API Free-Tier Rate Limit Reached: {e}. Please wait a moment before trying again.]",
                    provider=self.provider_name,
                    model=self.model_name,
                    is_error=True,
                    error_type="rate_limit"
                )
            elif "network" in err_str or "connection" in err_str or "timeout" in err_str:
                return AIResponse(
                    content=f"[Network Error connecting to Gemini API: {e}. Please check your internet connection.]",
                    provider=self.provider_name,
                    model=self.model_name,
                    is_error=True,
                    error_type="network_error"
                )
            else:
                return AIResponse(
                    content=f"[Gemini API Error: {e}]",
                    provider=self.provider_name,
                    model=self.model_name,
                    is_error=True,
                    error_type="api_error"
                )


class FreeLLMAPIProvider(BaseAIProvider):
    """FreeLLMAPI Unified Router Provider utilizing OpenAI-compatible REST endpoints."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None
    ):
        self.api_key = api_key or Config.FREELLMAPI_KEY
        self.base_url = (base_url or Config.FREELLMAPI_BASE_URL).rstrip("/")
        self.model_name = model or Config.FREELLMAPI_MODEL
        self.fallback_model = "gemini-3.1-flash-lite"  # Fast fallback if primary is down
        self.provider_name = "freellmapi"
        # Use full configured timeout for reading; fast 10s connect timeout
        read_timeout = timeout or Config.FREELLMAPI_TIMEOUT
        self.timeout = (10, read_timeout)  # (connect_timeout, read_timeout)
        self.max_retries = 1  # Retry once on network/timeout errors

    def is_available(self) -> bool:
        return bool(self.api_key and self.base_url)

    def _do_request(self, messages: List[Dict], headers: Dict, model: Optional[str] = None) -> "requests.Response":
        """Make the HTTP request with the configured timeout."""
        import requests
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model or self.model_name,
            "messages": messages,
            "temperature": 0.7,
            "stream": False,
        }
        return requests.post(url, headers=headers, json=payload, timeout=self.timeout)

    def generate(self, prompt: str, system_instruction: Optional[str] = None, model: Optional[str] = None) -> AIResponse:
        if not self.is_available():
            return AIResponse(
                content="[FreeLLMAPI key is not configured. Please add FREELLMAPI_KEY in the .env file.]",
                provider=self.provider_name,
                model=model or self.model_name,
                is_error=True,
                error_type="missing_api_key"
            )

        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        active_model = model or self.model_name
        last_error: Optional[str] = None

        for attempt in range(self.max_retries + 1):
            try:
                resp = self._do_request(messages, headers, model=active_model)

                if resp.status_code == 200:
                    data = resp.json()
                    choices = data.get("choices", [])
                    if choices and "message" in choices[0]:
                        content = choices[0]["message"].get("content", "").strip()
                        routed_model = data.get("model", active_model)
                        return AIResponse(
                            content=content,
                            provider=self.provider_name,
                            model=routed_model,
                            is_error=False,
                            raw=data
                        )
                    else:
                        return AIResponse(
                            content="[Empty response received from FreeLLMAPI router.]",
                            provider=self.provider_name,
                            model=active_model,
                            is_error=True,
                            error_type="empty_response"
                        )

                # If rate limited (429) or upstream error (502), try the fast fallback model
                if resp.status_code in (429, 502):
                    if active_model != self.fallback_model:
                        try:
                            fb_resp = self._do_request(messages, headers, model=self.fallback_model)
                            if fb_resp.status_code == 200:
                                fb_data = fb_resp.json()
                                fb_choices = fb_data.get("choices", [])
                                if fb_choices and "message" in fb_choices[0]:
                                    content = fb_choices[0]["message"].get("content", "").strip()
                                    return AIResponse(
                                        content=content,
                                        provider=self.provider_name,
                                        model=self.fallback_model,
                                        is_error=False,
                                        raw=fb_data
                                    )
                        except Exception:
                            pass

                    if resp.status_code == 429:
                        return AIResponse(
                            content="[FreeLLMAPI Rate Limit (429): Google AI Studio quota exceeded. Switching to a lighter model or waiting 30s is recommended.]",
                            provider=self.provider_name,
                            model=active_model,
                            is_error=True,
                            error_type="rate_limit"
                        )
                    return AIResponse(
                        content=f"[FreeLLMAPI: Primary model '{active_model}' is temporarily unavailable (502). Please try again in a moment.]",
                        provider=self.provider_name,
                        model=active_model,
                        is_error=True,
                        error_type="api_error"
                    )

                else:
                    return AIResponse(
                        content=f"[FreeLLMAPI Error {resp.status_code}: {resp.text[:300]}]",
                        provider=self.provider_name,
                        model=active_model,
                        is_error=True,
                        error_type="api_error"
                    )

            except Exception as e:
                err_str = str(e).lower()
                last_error = str(e)
                is_timeout = "timeout" in err_str or "timed out" in err_str or "read timeout" in err_str
                is_conn = "connection" in err_str or "refused" in err_str or "unreachable" in err_str

                if (is_timeout or is_conn) and attempt < self.max_retries:
                    # Try fallback model immediately on timeout
                    if active_model != self.fallback_model:
                        active_model = self.fallback_model
                    time.sleep(1)
                    continue

                if is_timeout:
                    return AIResponse(
                        content=(
                            f"[Request timed out waiting for FreeLLMAPI (model: {self.model_name}). "
                            f"The model took too long to respond. Try again or use '/provider mock'.]"
                        ),
                        provider=self.provider_name,
                        model=self.model_name,
                        is_error=True,
                        error_type="network_error"
                    )
                elif is_conn:
                    return AIResponse(
                        content=(
                            f"[Cannot connect to FreeLLMAPI at {self.base_url}. "
                            f"Make sure 'npm run dev' is running in C:\\Users\\LENOVO\\freellmapi. Details: {e}]"
                        ),
                        provider=self.provider_name,
                        model=self.model_name,
                        is_error=True,
                        error_type="network_error"
                    )
                return AIResponse(
                    content=f"[FreeLLMAPI Request Error: {e}]",
                    provider=self.provider_name,
                    model=self.model_name,
                    is_error=True,
                    error_type="api_error"
                )

        return AIResponse(
            content=f"[FreeLLMAPI: All {self.max_retries + 1} attempts failed. Last error: {last_error}]",
            provider=self.provider_name,
            model=self.model_name,
            is_error=True,
            error_type="network_error"
        )


class AIClient:
    """Unified client that routes between FreeLLMAPI, Gemini, and Mock providers seamlessly."""

    def __init__(self, provider_type: Optional[str] = None):
        self.provider_type = (provider_type or Config.AI_PROVIDER).lower()
        self.mock_provider = MockAIProvider()
        self.gemini_provider = GeminiAIProvider()
        self.freellmapi_provider = FreeLLMAPIProvider()

    def get_active_provider(self) -> BaseAIProvider:
        """Select active provider based on configuration and key availability."""
        if self.provider_type == "mock":
            return self.mock_provider

        if self.provider_type == "freellmapi":
            return self.freellmapi_provider

        if self.provider_type == "gemini":
            return self.gemini_provider

        # Auto mode:
        # 1. Prefer FreeLLMAPI if configured and available
        if self.freellmapi_provider.is_available():
            return self.freellmapi_provider

        # 2. Try Gemini if configured and available
        if self.gemini_provider.is_available():
            return self.gemini_provider

        # 3. Fallback to deterministic Mock
        return self.mock_provider

    def generate(self, prompt: str, system_instruction: Optional[str] = None, model: Optional[str] = None) -> AIResponse:
        """Generate response via active provider with graceful fallback."""
        provider = self.get_active_provider()
        if hasattr(provider, "generate") and "model" in provider.generate.__code__.co_varnames:
            resp = provider.generate(prompt, system_instruction=system_instruction, model=model)
        else:
            resp = provider.generate(prompt, system_instruction=system_instruction)

        # If live provider returned missing API key or connection error and we are in auto mode, fallback to mock
        if resp.is_error and resp.error_type in ("missing_api_key", "network_error") and self.provider_type == "auto":
            return self.mock_provider.generate(prompt, system_instruction=system_instruction)

        return resp
