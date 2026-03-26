from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    serpapi_api_key: str = ""
    openai_api_key: str = ""
    openrouter_api_key: str = ""
    llm_model: str = "gpt-4o"
    openrouter_model: str = "anthropic/claude-sonnet-4-20250514"
    default_origin: str = "Bangalore"
    default_currency: str = "INR"
    db_path: str = "travel_history.db"
    cache_ttl_seconds: int = 86400
    api_timeout_seconds: int = 10
    api_max_retries: int = 3
    tavily_api_key: str = ""
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "wandermust-travel-optimizer/1.0"
    social_extraction_model: str = "gpt-4o-mini"
    discovery_model: str = "gpt-4o"
    max_onboarding_questions: int = 5
    max_discovery_questions: int = 5

    # Discovery v2
    discovery_v2_model: str = "gpt-4o"
    discovery_v2_min_profile_turns: int = 2
    discovery_v2_min_discovery_turns: int = 2
    discovery_v2_min_narrowing_turns: int = 1

    class Config:
        env_file = ".env"

settings = Settings()
