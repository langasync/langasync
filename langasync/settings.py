from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LangasyncSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", populate_by_name=True, extra="ignore"
    )

    # Provider API keys — use standard env var names (no prefix)
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1", validation_alias="OPENAI_BASE_URL"
    )
    anthropic_base_url: str = Field(
        default="https://api.anthropic.com", validation_alias="ANTHROPIC_BASE_URL"
    )

    # Langasync-specific settings — use LANGASYNC_ prefix
    batch_poll_interval: float = Field(
        default=60.0, validation_alias="LANGASYNC_BATCH_POLL_INTERVAL"
    )
    base_storage_path: str = Field(
        default="./langasync_jobs", validation_alias="LANGASYNC_BASE_STORAGE_PATH"
    )


langasync_settings = LangasyncSettings()
