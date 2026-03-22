"""Shared LLM factory — supports OpenAI directly or OpenRouter as fallback.

If OPENAI_API_KEY is set, uses OpenAI API directly (cheaper, no proxy).
If only OPENROUTER_API_KEY is set, uses OpenRouter (multi-model gateway).
"""
from langchain_openai import ChatOpenAI
from config import settings


def get_llm(model: str = ""):
    """Create a ChatOpenAI instance using the best available provider.

    Args:
        model: Model name override. If empty, uses settings.llm_model (for OpenAI)
               or settings.openrouter_model (for OpenRouter).
    """
    if settings.openai_api_key:
        return ChatOpenAI(
            model=model or settings.llm_model,
            api_key=settings.openai_api_key,
        )
    elif settings.openrouter_api_key:
        # Map OpenAI model names to OpenRouter equivalents if needed
        or_model = model or settings.openrouter_model
        return ChatOpenAI(
            model=or_model,
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
        )
    else:
        raise ValueError(
            "No LLM API key configured. Set OPENAI_API_KEY or OPENROUTER_API_KEY in .env"
        )
