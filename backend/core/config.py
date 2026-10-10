from enum import Enum
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ComputeProfile(str, Enum):
    POTATO = "POTATO"  # 8GB RAM / CPU
    HYBRID = "HYBRID"  # Cloud API / Colab
    BEAST = "BEAST"    # Local RTX 3060 12GB CUDA Rig


class LLMProvider(str, Enum):
    OLLAMA = "OLLAMA"  # Local Qwen3.5:9B via Ollama
    GEMINI = "GEMINI"  # Google AI Studio Gemini 3.8 Flash


class Settings(BaseSettings):
    COMPUTE_PROFILE: ComputeProfile = Field(
        default=ComputeProfile.BEAST,
        description="Active hardware execution profile"
    )

    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/dealersflow_dev",
        description="Async PostgreSQL connection string"
    )

    # Local Ollama Settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LOCAL_LLM_MODEL: str = "qwen3.5:9b"

    # Granular Engine Selection
    INTAKE_LLM_PROVIDER: LLMProvider = Field(
        default=LLMProvider.OLLAMA,
        description="Provider for Voice Notes & Single Text Intake"
    )
    BATCH_LLM_PROVIDER: LLMProvider = Field(
        default=LLMProvider.GEMINI,
        description="Provider for 100-Message WhatsApp Group Chat Dumps"
    )

    # Google AI Studio Settings
    GEMINI_API_KEY: Optional[str] = Field(
        default=None,
        description="Google AI Studio API Key"
    )
    GEMINI_MODEL: str = Field(
        default="gemini-3.8-flash",
        description="Google Gemini model identifier"
    )

    COLAB_ENDPOINT_URL: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Defensive Backward-Compatibility Property Alias
    @property
    def LLM_PROVIDER(self) -> LLMProvider:
        """Alias returning active intake provider to prevent legacy AttributeError."""
        return self.INTAKE_LLM_PROVIDER

    @property
    def whisper_model_size(self) -> str:
        if self.COMPUTE_PROFILE == ComputeProfile.POTATO:
            return "small"
        return "large-v3-turbo"

    @property
    def whisper_device(self) -> str:
        if self.COMPUTE_PROFILE == ComputeProfile.POTATO:
            return "cpu"
        return "cuda"

    @property
    def whisper_compute_type(self) -> str:
        if self.COMPUTE_PROFILE == ComputeProfile.POTATO:
            return "int8"
        return "float16"

    @property
    def active_llm_base_url(self) -> str:
        if self.COMPUTE_PROFILE == ComputeProfile.HYBRID and self.COLAB_ENDPOINT_URL:
            return f"{self.COLAB_ENDPOINT_URL.rstrip('/')}/v1"
        return f"{self.OLLAMA_BASE_URL.rstrip('/')}/v1"

    @property
    def active_llm_model(self) -> str:
        if self.INTAKE_LLM_PROVIDER == LLMProvider.GEMINI:
            return self.GEMINI_MODEL
        if self.COMPUTE_PROFILE == ComputeProfile.POTATO:
            return "qwen3:1.7b"
        return self.LOCAL_LLM_MODEL


settings = Settings()