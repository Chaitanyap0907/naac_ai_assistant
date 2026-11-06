"""
app/services/ibm_granite.py  —  IBM Granite LLM wrapper
Lazy-imports the IBM SDK so the app starts even if ibm-watsonx-ai
is not installed (graceful degradation to context-synthesis mode).
"""
from __future__ import annotations
from typing import Optional
from app.core import get_settings, logger

_settings = get_settings()
_PLACEHOLDER_KEYS = {"", "your_ibm_cloud_api_key_here", "your_watsonx_project_id_here"}


def _credentials_configured() -> bool:
    return (
        _settings.ibm_cloud_api_key not in _PLACEHOLDER_KEYS
        and _settings.ibm_project_id   not in _PLACEHOLDER_KEYS
    )


def _try_import_sdk():
    """Lazy import so missing ibm-watsonx-ai doesn't crash startup."""
    try:
        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference
        from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
        return Credentials, ModelInference, GenParams
    except ImportError as e:
        logger.warning(f"ibm-watsonx-ai not available ({e}) — fallback mode active")
        return None, None, None


class GraniteClient:
    """
    Thin wrapper around IBM Granite (watsonx.ai).
    Falls back to local context-synthesis when SDK or credentials are absent.
    """

    def __init__(
        self,
        model_id:    Optional[str]   = None,
        max_tokens:  Optional[int]   = None,
        temperature: Optional[float] = None,
    ):
        self.model_id    = model_id    or _settings.granite_model_id
        self.max_tokens  = max_tokens  or _settings.rag_max_tokens
        self.temperature = temperature or _settings.rag_temperature
        self._model      = None

        Credentials, ModelInference, GenParams = _try_import_sdk()
        self._sdk_available = Credentials is not None
        self._use_fallback  = (not self._sdk_available) or (not _credentials_configured())

        if self._use_fallback:
            reason = "SDK not installed" if not self._sdk_available else "credentials not configured"
            logger.warning(f"GraniteClient: {reason} — LOCAL FALLBACK mode active")

    # ── Internal: build model lazily ─────────────────────────────────────────
    def _get_model(self):
        if self._model is None:
            Credentials, ModelInference, GenParams = _try_import_sdk()
            if Credentials is None:
                raise RuntimeError("ibm-watsonx-ai SDK not installed")
            logger.info(f"Initialising IBM Granite model: {self.model_id}")
            self._model = ModelInference(
                model_id=self.model_id,
                credentials=Credentials(
                    url=_settings.ibm_cloud_url,
                    api_key=_settings.ibm_cloud_api_key,
                ),
                project_id=_settings.ibm_project_id,
                params={
                    GenParams.MAX_NEW_TOKENS:    self.max_tokens,
                    GenParams.TEMPERATURE:       self.temperature,
                    GenParams.REPETITION_PENALTY: 1.1,
                    GenParams.STOP_SEQUENCES:    ["</s>", "\n\nHuman:", "\n\nUser:"],
                },
            )
        return self._model

    # ── Public API ────────────────────────────────────────────────────────────
    def generate(self, prompt: str, context_only: str = "") -> str:
        if self._use_fallback:
            return _local_synthesize(prompt, context_only)
        return self._granite_generate(prompt, context_only)

    def _granite_generate(self, prompt: str, context_only: str = "") -> str:
        try:
            model = self._get_model()
            logger.debug(f"Sending prompt ({len(prompt)} chars) to {self.model_id}")
            response = model.generate_text(prompt=prompt)
            return response.strip() if isinstance(response, str) else str(response)
        except Exception as exc:
            logger.error(f"IBM Granite error: {exc} — falling back to context synthesis")
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
            yield _local_synthesize(prompt, "")


# ── Local fallback synthesizer ────────────────────────────────────────────────
def _local_synthesize(prompt: str, context: str) -> str:
    if not context or context.startswith("No relevant context"):
        return (
            "No relevant information was found in the NAAC knowledge base.\n\n"
            "Please upload additional NAAC documents via the Upload panel, "
            "or configure IBM Cloud credentials in Streamlit Secrets to enable IBM Granite AI."
        )
    clean_chunks = []
    for chunk in context.split("\n\n"):
        text = chunk.strip()
        if text.startswith("[Source") and "]" in text:
            text = text[text.index("]") + 1:].strip()
        if text:
            clean_chunks.append(text)
    combined = "\n\n".join(clean_chunks)
    if len(combined) > 1500:
        combined = combined[:1500].rsplit(".", 1)[0] + "."
    return (
        "**From the NAAC Knowledge Base** *(IBM Granite fallback — configure credentials for AI responses)*:\n\n"
        + combined
        + "\n\n---\n*Set IBM_CLOUD_API_KEY & IBM_PROJECT_ID in Streamlit Secrets to enable full AI responses.*"
    )


# ── Singleton ─────────────────────────────────────────────────────────────────
_client: Optional[GraniteClient] = None

def get_granite_client() -> GraniteClient:
    global _client
    if _client is None:
        _client = GraniteClient()
    return _client
