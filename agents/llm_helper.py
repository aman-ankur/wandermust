"""Shared LLM factory — supports OpenAI directly or OpenRouter as fallback.

If OPENAI_API_KEY is set, uses OpenAI API directly (cheaper, no proxy).
If only OPENROUTER_API_KEY is set, uses OpenRouter (multi-model gateway).
"""
import json
import logging
import re
from langchain_openai import ChatOpenAI
from config import settings

logger = logging.getLogger("wandermust.llm")


def parse_json_response(content: str):
    """Parse JSON from LLM response, stripping markdown code fences if present."""
    text = content.strip()
    # Strip ```json ... ``` or ``` ... ```
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    return json.loads(text)


def get_llm(model: str = ""):
    """Create a ChatOpenAI instance using the best available provider.

    Args:
        model: Model name override. If empty, uses settings.llm_model (for OpenAI)
               or settings.openrouter_model (for OpenRouter).
    """
    if settings.openai_api_key:
        resolved = model or settings.llm_model
        logger.info(f"Creating LLM via OpenAI — model={resolved}")
        return ChatOpenAI(
            model=resolved,
            api_key=settings.openai_api_key,
        )
    elif settings.openrouter_api_key:
        or_model = model or settings.openrouter_model
        logger.info(f"Creating LLM via OpenRouter — model={or_model}")
        return ChatOpenAI(
            model=or_model,
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
        )
    else:
        logger.error("No LLM API key configured")
        raise ValueError(
            "No LLM API key configured. Set OPENAI_API_KEY or OPENROUTER_API_KEY in .env"
        )
