from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    MODEL_PATH: Optional[str] = None
    MAX_TOKENS: int = 2000
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.95

    class Config:
        env_file = ".env"

settings = Settings()
