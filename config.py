from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    amadeus_client_id: str = ""
    amadeus_client_secret: str = ""
    openrouter_api_key: str = ""
    openrouter_model: str = "anthropic/claude-sonnet-4-20250514"
    default_origin: str = "Bangalore"
    default_currency: str = "INR"
    db_path: str = "travel_history.db"
    cache_ttl_seconds: int = 86400
    api_timeout_seconds: int = 10
    api_max_retries: int = 3

    class Config:
        env_file = ".env"

settings = Settings()
