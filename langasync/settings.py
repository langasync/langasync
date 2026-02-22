from pydantic import AliasChoices, Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

_BEDROCK_REGION_GEO_PREFIXES = {
    "us": "us",
    "eu": "eu",
    "ap": "apac",
    "ca": "us",
    "me": "apac",
    "il": "eu",
}


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
    google_api_key: str | None = Field(default=None, validation_alias="GOOGLE_API_KEY")
    google_base_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta",
        validation_alias="GOOGLE_BASE_URL",
    )
    aws_access_key_id: str | None = Field(default=None, validation_alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str | None = Field(
        default=None, validation_alias="AWS_SECRET_ACCESS_KEY"
    )
    aws_session_token: str | None = Field(default=None, validation_alias="AWS_SESSION_TOKEN")
    aws_region: str | None = Field(
        default=None,
        validation_alias=AliasChoices("AWS_REGION", "AWS_DEFAULT_REGION"),
    )
    bedrock_s3_bucket: str | None = Field(default=None, validation_alias="BEDROCK_S3_BUCKET")
    bedrock_role_arn: str | None = Field(default=None, validation_alias="BEDROCK_ROLE_ARN")
    bedrock_base_url: str | None = Field(default=None, validation_alias="BEDROCK_BASE_URL")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def bedrock_region_prefix(self) -> str | None:
        """Cross-region inference profile prefix derived from aws_region (e.g. 'us', 'eu', 'apac')."""
        if not self.aws_region:
            return None
        geo = self.aws_region.split("-")[0]
        if geo not in _BEDROCK_REGION_GEO_PREFIXES:
            raise ValueError(
                f"Cannot determine Bedrock cross-region prefix for region '{self.aws_region}'. "
                f"Supported region prefixes: {sorted(_BEDROCK_REGION_GEO_PREFIXES.keys())}"
            )
        return _BEDROCK_REGION_GEO_PREFIXES[geo]

    # Langasync-specific settings — use LANGASYNC_ prefix
    batch_poll_interval: float = Field(
        default=60.0, validation_alias="LANGASYNC_BATCH_POLL_INTERVAL"
    )
    base_storage_path: str = Field(
        default="./langasync_jobs", validation_alias="LANGASYNC_BASE_STORAGE_PATH"
    )


langasync_settings = LangasyncSettings()
