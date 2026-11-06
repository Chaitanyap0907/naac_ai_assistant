"""
app/services/ibm_granite.py
IBM watsonx.ai Granite LLM wrapper.

Uses ibm-watsonx-ai SDK to call IBM Granite models (granite-13b-chat-v2)
available on the IBM Cloud Lite (free) tier.

If IBM credentials are not configured (placeholder values), the client
automatically falls back to a local context-synthesis mode so the app
remains fully usable without a watsonx.ai account.
"""
from __future__ import annotations

from typing import Optional

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core import get_settings, logger

_settings = get_settings()

_PLACEHOLDER_KEYS = {"", "your_ibm_cloud_api_key_here", "your_watsonx_project_id_here"}


def _credentials_configured() -> bool:
    """Return True only if real (non-placeholder) IBM credentials are present."""
    return (
        _settings.ibm_cloud_api_key not in _PLACEHOLDER_KEYS
        and _settings.ibm_project_id not in _PLACEHOLDER_KEYS
    )


def _build_credentials() -> Credentials:
    return Credentials(
        url=_settings.ibm_cloud_url,
        api_key=_settings.ibm_cloud_api_key,
    )


class GraniteClient:
    """
    Thin wrapper around IBM Granite models via watsonx.ai SDK.
    Falls back to local context-synthesis when credentials are absent.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ):
        self.model_id = model_id or _settings.granite_model_id
        self.max_tokens = max_tokens or _settings.rag_max_tokens
        self.temperature = temperature or _settings.rag_temperature
        self._model: Optional[ModelInference] = None
        self._use_fallback: bool = not _credentials_configured()

        if self._use_fallback:
            logger.warning(
                "IBM Cloud credentials not configured — running in LOCAL FALLBACK mode. "
                "Set IBM_CLOUD_API_KEY and IBM_PROJECT_ID in .env to enable IBM Granite."
            )

    def _get_model(self) -> ModelInference:
        if self._model is None:
            logger.info(f"Initialising IBM Granite model: {self.model_id}")
            self._model = ModelInference(
                model_id=self.model_id,
                credentials=_build_credentials(),
                project_id=_settings.ibm_project_id,
                params={
                    GenParams.MAX_NEW_TOKENS: self.max_tokens,
                    GenParams.TEMPERATURE: self.temperature,
                    GenParams.REPETITION_PENALTY: 1.1,
                    GenParams.STOP_SEQUENCES: ["</s>", "\n\nHuman:", "\n\nUser:"],
                },
            )
        return self._model

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(self, prompt: str, context_only: str = "") -> str:
        """
        Generate text.
        - If IBM credentials are set: calls watsonx.ai Granite.
        - Otherwise: returns a structured synthesis built from context_only.
        """
        if self._use_fallback:
            return _local_synthesize(prompt, context_only)
        return self._granite_generate(prompt, context_only)

    def _granite_generate(self, prompt: str, context_only: str = "") -> str:
        try:
            model = self._get_model()
            logger.debug(f"Sending prompt ({len(prompt)} chars) to {self.model_id}")
            response = model.generate_text(prompt=prompt)
            logger.debug("Received response from IBM Granite")
            return response.strip() if isinstance(response, str) else str(response)
        except Exception as exc:
            logger.error(f"IBM Granite generation error: {exc} — falling back to context synthesis")
            return _local_synthesize(prompt, context_only)

    def generate_stream(self, prompt: str):
        if self._use_fallback:
            yield _local_synthesize(prompt, "")
            return
        try:
            model = self._get_model()
            for chunk in model.generate_text_stream(prompt=prompt):
                yield chunk
        except Exception as exc:
            logger.error(f"IBM Granite stream error: {exc}")
            raise


# ── Local fallback synthesizer ────────────────────────────────────────────────

def _local_synthesize(prompt: str, context: str) -> str:
    """
    Build a structured NAAC answer directly from the retrieved context
    without an external LLM call.  Works offline, zero latency.
    """
    if not context or context.startswith("No relevant context"):
        return (
            "No relevant information was found in the NAAC knowledge base for your query.\n\n"
            "Please upload additional NAAC guideline documents via the 'Upload Document' panel, "
            "or configure IBM Cloud credentials in the .env file to enable IBM Granite AI responses."
        )

    # Extract the question/task from the prompt (last non-empty line before "### Answer:" etc.)
    lines = [l.strip() for l in prompt.split("\n") if l.strip()]
    task_line = ""
    for i, line in enumerate(lines):
        if line.startswith("### Faculty Question:") or line.startswith("### Task:"):
            if i + 1 < len(lines):
                task_line = lines[i + 1]
            break

    # Strip source tags from context for cleaner output
    clean_chunks = []
    for chunk in context.split("\n\n"):
        text = chunk.strip()
        if text.startswith("[Source"):
            text = text[text.index("]") + 1:].strip() if "]" in text else text
        if text:
            clean_chunks.append(text)

    combined = "\n\n".join(clean_chunks)

    header = (
        f"**Based on the NAAC Knowledge Base** "
        f"({'IBM Granite fallback — add credentials for AI-generated responses'}):\n\n"
    )

    # Keep response under ~1200 chars for readability
    if len(combined) > 1200:
        combined = combined[:1200].rsplit(".", 1)[0] + "."

    return header + combined + (
        "\n\n---\n*To get AI-generated, context-aware responses, "
        "set IBM_CLOUD_API_KEY and IBM_PROJECT_ID in your .env file.*"
    )


# ── Module-level singleton ─────────────────────────────────────────────────────
_granite_client: Optional[GraniteClient] = None


def get_granite_client() -> GraniteClient:
    global _granite_client
    if _granite_client is None:
        _granite_client = GraniteClient()
    return _granite_client
