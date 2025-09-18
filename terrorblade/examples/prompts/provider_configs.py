#!/usr/bin/env python3
"""
Provider and Model Configurations for Terrorblade LLM Integration

Defines available provider:model combinations with latest 2024-2025 models
used across the Terrorblade ecosystem, including analyze_dialogues.py and
multi_provider_benchmark.py.

Updated with latest models as of 2025:
- OpenAI: GPT-4.1 series (latest), GPT-4o
- Anthropic Claude: 3.5 Sonnet/Haiku, 4.0 series
- Google Gemini: 2.5 Pro/Flash (latest)
- Mistral: Large, Small v3.1, Saba
- CognitiveComputations Dolphin: Latest uncensored models
"""

from typing import Dict, List

# Provider:Model combinations with latest 2024-2025 models
PROVIDER_MODEL_COMBINATIONS = {
    "openai": {
        "models": [
            "gpt-4.1",                    # Latest 2025 - best performance
            "gpt-4.1-mini",              # Latest 2025 - fast, cost-effective
            "gpt-4.1-nano",              # Latest 2025 - fastest, cheapest
            "gpt-4o",                     # Multimodal flagship
            "gpt-4o-mini",               # Cost-effective workhorse
            "gpt-4-turbo",               # Previous generation high-performance
        ],
        "env_key": "OPENAI_API_KEY",
        "display_name": "OpenAI",
        "default_model": "gpt-4.1-mini"
    },
    "openrouter": {
        "models": [
            # Anthropic Claude (via OpenRouter)
            "anthropic/claude-3.5-sonnet",    # Current best Claude
            "anthropic/claude-3.5-haiku",     # Fast Claude
            "anthropic/claude-3-opus",        # Previous flagship
            "anthropic/claude-3-haiku",       # Previous fast model

            # Google Gemini (via OpenRouter)
            "google/gemini-2.5-pro",          # Latest advanced reasoning
            "google/gemini-2.5-flash",        # Latest fast model
            "google/gemini-2.0-flash",        # Previous generation
            "google/gemini-pro",              # Legacy model

            # Mistral (via OpenRouter)
            "mistralai/mistral-large",        # Top-tier reasoning
            "mistralai/mistral-small",        # Refined intermediate
            "mistralai/pixtral-large",        # Multimodal

            # Meta Llama
            "meta-llama/llama-3.1-8b-instruct",
            "meta-llama/llama-3.1-70b-instruct",

            # Dolphin (Uncensored)
            "cognitivecomputations/dolphin-mixtral-8x7b",
        ],
        "env_key": "OPENROUTER_API_KEY",
        "display_name": "OpenRouter",
        "default_model": "google/gemini-2.5-pro"
    },
    "deepinfra": {
        "models": [
            # Meta Llama
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "meta-llama/Meta-Llama-3.1-70B-Instruct",
            "meta-llama/Meta-Llama-3.1-405B-Instruct",

            # Mistral
            "mistralai/Mistral-7B-Instruct-v0.3",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "mistralai/Mixtral-8x22B-Instruct-v0.1",

            # Dolphin (Uncensored)
            "cognitivecomputations/dolphin-2.8-mistral-7b-v02",
            "cognitivecomputations/dolphin-mixtral-8x7b",

            # Other models
            "microsoft/DialoGPT-medium",
        ],
        "env_key": "DEEPINFRA_API_KEY",
        "display_name": "DeepInfra",
        "default_model": "meta-llama/Meta-Llama-3.1-8B-Instruct"
    },
    "fireworks": {
        "models": [
            # Llama models
            "accounts/fireworks/models/llama-v3p1-8b-instruct",
            "accounts/fireworks/models/llama-v3p1-70b-instruct",
            "accounts/fireworks/models/llama-v3p1-405b-instruct",

            # Mixtral models
            "accounts/fireworks/models/mixtral-8x7b-instruct",
            "accounts/fireworks/models/mixtral-8x22b-instruct",

            # Qwen models
            "accounts/fireworks/models/qwen2p5-72b-instruct",
        ],
        "env_key": "FIREWORKS_API_KEY",
        "display_name": "Fireworks",
        "default_model": "accounts/fireworks/models/llama-v3p1-8b-instruct"
    },
    "anthropic": {
        "models": [
            "claude-3-5-sonnet-20241022",    # Latest Claude 3.5 Sonnet
            "claude-3-5-haiku-20241022",     # Latest Claude 3.5 Haiku
            "claude-3-opus-20240229",       # Claude 3 Opus
            "claude-3-sonnet-20240229",     # Claude 3 Sonnet
            "claude-3-haiku-20240307",      # Claude 3 Haiku
        ],
        "env_key": "ANTHROPIC_API_KEY",
        "display_name": "Anthropic",
        "default_model": "claude-3-5-sonnet-20241022"
    },
    "google": {
        "models": [
            "gemini-2.5-pro",               # Latest advanced reasoning
            "gemini-2.5-flash",             # Latest fast model
            "gemini-2.0-flash",             # Previous generation
            "gemini-1.5-pro",              # Legacy pro model
            "gemini-1.5-flash",            # Legacy flash model
        ],
        "env_key": "GOOGLE_API_KEY",
        "display_name": "Google",
        "default_model": "gemini-2.5-flash"
    }
}

def get_available_prompts() -> List[str]:
    """Get list of available prompt files."""
    from pathlib import Path
    prompts_dir = Path(__file__).parent
    return [f.name for f in prompts_dir.glob("*.md")]

def get_provider_models() -> Dict[str, List[str]]:
    """Get simplified provider:models mapping."""
    return {provider: config["models"] for provider, config in PROVIDER_MODEL_COMBINATIONS.items()}

def get_all_provider_model_pairs() -> List[tuple[str, str]]:
    """Get all possible provider:model combinations as tuples."""
    pairs = []
    for provider, config in PROVIDER_MODEL_COMBINATIONS.items():
        for model in config["models"]:
            pairs.append((provider, model))
    return pairs

def get_default_provider_model_pairs() -> List[tuple[str, str]]:
    """Get default model for each provider."""
    return [(provider, config["default_model"])
            for provider, config in PROVIDER_MODEL_COMBINATIONS.items()
            if "default_model" in config]

def format_provider_model_display(provider: str, model: str) -> str:
    """Format provider:model for display purposes."""
    display_name = PROVIDER_MODEL_COMBINATIONS.get(provider, {}).get("display_name", provider.capitalize())

    # Shorten long model names for display
    if "/" in model:
        model_display = model.split("/")[-1]
    elif model.startswith("accounts/fireworks/models/"):
        model_display = model.replace("accounts/fireworks/models/", "")
    elif model.startswith("claude-"):
        # Clean up Claude model names
        model_display = model.replace("-20240229", "").replace("-20240307", "").replace("-20241022", "")
    elif model.startswith("gemini-"):
        model_display = model
    else:
        model_display = model

    return f"{display_name}: {model_display}"

def validate_provider_model(provider: str, model: str) -> bool:
    """Validate if provider:model combination is supported."""
    if provider not in PROVIDER_MODEL_COMBINATIONS:
        return False
    return model in PROVIDER_MODEL_COMBINATIONS[provider]["models"]

def get_recommended_models() -> Dict[str, str]:
    """Get recommended models for different use cases."""
    return {
        "best_performance": "gpt-4.1",
        "cost_effective": "gpt-4.1-mini",
        "fastest": "gpt-4.1-nano",
        "multimodal": "gpt-4o",
        "reasoning": "gemini-2.5-pro",
        "uncensored": "cognitivecomputations/dolphin-2.8-mistral-7b-v02",
        "claude_latest": "claude-3-5-sonnet-20241022",
        "open_source": "meta-llama/Meta-Llama-3.1-70B-Instruct"
    }

def get_providers_with_env_keys() -> Dict[str, str]:
    """Get mapping of providers to their environment variable names."""
    return {provider: config["env_key"]
            for provider, config in PROVIDER_MODEL_COMBINATIONS.items()}