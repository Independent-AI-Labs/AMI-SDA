# sda/core/config_models.py

"""
Defines the Pydantic models for all AI-related configurations.

This module provides a structured, type-safe definition for the configuration
of Large Language Models (LLMs) and Embedding Models, separating the configuration
schema from its instantiation in the main config file. This prevents circular imports.
"""

from typing import List, Dict
from pydantic import BaseModel, Field


class LLMRateLimit(BaseModel):
    """Defines a single rate limit (e.g., 30 requests per minute)."""
    requests: float
    period_seconds: float


class LLMConfig(BaseModel):
    """A structured configuration for a single Large Language Model."""
    model_name: str = Field(..., description="The identifier for the model (e.g., 'gemini-1.5-pro').")
    max_context_tokens: int = Field(..., description="The maximum number of tokens allowed in the model's context window.")
    soft_context_limit_tokens: int = Field(..., description="A soft limit to trigger warnings before hitting the max context.")
    rate_limits: List[LLMRateLimit] = Field(..., description="A list of rate limits to apply to this model.")
    input_price_per_million_tokens: float = Field(0.0, description="Cost for 1 million input tokens.")
    output_price_per_million_tokens: float = Field(0.0, description="Cost for 1 million output tokens.")
    provider: str = Field(..., description="The API provider (e.g., 'google', 'openai', 'anthropic').")


class EmbeddingConfig(BaseModel):
    """A structured configuration for a single Embedding Model."""
    model_name: str = Field(..., description="The identifier for the model (e.g., 'jina-embedding-v2-base-en').")
    max_tokens: int = Field(..., description="The maximum number of tokens for a single embedding input.")
    dimension: int = Field(..., description="The dimensionality of the output embedding vector.")
    price_per_million_tokens: float = Field(..., description="Cost for processing 1 million tokens.")
    provider: str = Field(..., description="The provider or hosting type (e.g., 'local', 'openai').")


class AIProviderConfig(BaseModel):
    """Groups all model configurations by their provider."""
    llms: Dict[str, LLMConfig] = Field(default_factory=dict, description="Configurations for Language Models.")
    embeddings: Dict[str, EmbeddingConfig] = Field(default_factory=dict, description="Configurations for Embedding Models.")


class AIConfigModel(BaseModel):
    """The root model for all AI configurations."""
    google: AIProviderConfig = Field(default_factory=AIProviderConfig)
    openai: AIProviderConfig = Field(default_factory=AIProviderConfig)
    anthropic: AIProviderConfig = Field(default_factory=AIProviderConfig)
    local: AIProviderConfig = Field(default_factory=AIProviderConfig, description="For self-hosted models.")