#!/usr/bin/env python3
"""
Promptinator - Universal LLM Provider Interface for Terrorblade

Handles multiple LLM providers (OpenAI, OpenRouter, DeepInfra, Fireworks) with
automatic provider detection, failover, and prompt management.

Usage:
    promptinator = Promptinator(provider="openai")
    response = promptinator.query("Analyze this text", prompt_file="prompt_1.md")
"""

import os
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import openai
import requests
from dotenv import load_dotenv

load_dotenv()


class LLMProvider(Enum):
    """Available LLM providers."""
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    DEEPINFRA = "deepinfra"
    FIREWORKS = "fireworks"


@dataclass
class ProviderConfig:
    """Configuration for each LLM provider."""
    name: str
    base_url: str
    api_key_env: str
    default_model: str
    chat_model: str
    ping_endpoint: str
    headers: dict[str, str] | None = None


@dataclass
class LLMResponse:
    """Standardized response from LLM providers."""
    content: str
    provider: str
    model: str
    tokens_used: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost: float | None = None
    error: str | None = None


class Promptinator:
    """Universal LLM provider interface with prompt management."""

    # Provider configurations
    PROVIDERS = {
        LLMProvider.OPENAI: ProviderConfig(
            name="OpenAI",
            base_url="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
            default_model="gpt-4o-mini",
            chat_model="gpt-4o-mini",
            ping_endpoint="https://api.openai.com/v1/models"
        ),
        LLMProvider.OPENROUTER: ProviderConfig(
            name="OpenRouter",
            base_url="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
            default_model="anthropic/claude-3.5-sonnet",
            chat_model="anthropic/claude-3.5-sonnet",
            ping_endpoint="https://openrouter.ai/api/v1/models",
            headers={"HTTP-Referer": "https://github.com/terrorblade", "X-Title": "Terrorblade"}
        ),
        LLMProvider.DEEPINFRA: ProviderConfig(
            name="DeepInfra",
            base_url="https://api.deepinfra.com/v1/openai",
            api_key_env="DEEPINFRA_API_KEY",
            default_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            chat_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            ping_endpoint="https://api.deepinfra.com/v1/openai/models"
        ),
        LLMProvider.FIREWORKS: ProviderConfig(
            name="Fireworks",
            base_url="https://api.fireworks.ai/inference/v1",
            api_key_env="FIREWORKS_API_KEY",
            default_model="accounts/fireworks/models/llama-v3p1-8b-instruct",
            chat_model="accounts/fireworks/models/llama-v3p1-8b-instruct",
            ping_endpoint="https://api.fireworks.ai/inference/v1/models"
        )
    }

    def __init__(self, provider: str = "openai", model: str | None = None):
        """
        Initialize Promptinator with specified provider.

        Args:
            provider: Provider name (openai, openrouter, deepinfra, fireworks)
            model: Optional model override
        """
        self.provider_enum = LLMProvider(provider.lower())
        self.config = self.PROVIDERS[self.provider_enum]
        self.model = model or self.config.default_model
        self.prompts_dir = Path(__file__).parent

        # Get API key
        self.api_key = os.getenv(self.config.api_key_env)
        if not self.api_key:
            raise ValueError(f"API key not found in environment: {self.config.api_key_env}")

        # Initialize OpenAI client with provider configuration
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.config.base_url,
            default_headers=self.config.headers or {}
        )

        # Validate connection
        if not self._ping_provider():
            raise ConnectionError(f"Cannot connect to {self.config.name}")

    def _ping_provider(self) -> bool:
        """Test provider connectivity."""
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            if self.config.headers:
                headers.update(self.config.headers)

            response = requests.get(
                self.config.ping_endpoint,
                headers=headers,
                timeout=10
            )
            return response.status_code == 200
        except Exception:
            return False

    def get_available_models(self) -> list[str]:
        """Get list of available models from current provider."""
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            print(f"Error fetching models: {e}")
            return [self.config.default_model, self.config.chat_model]

    def switch_provider(self, provider: str, model: str | None = None) -> None:
        """Switch to a different provider."""
        self.__init__(provider, model)

    def load_prompt(self, prompt_file: str) -> str:
        """Load prompt from prompts directory."""
        prompt_path = self.prompts_dir / prompt_file
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

        return prompt_path.read_text(encoding='utf-8')

    def list_available_prompts(self) -> list[str]:
        """List all available prompt files."""
        return [f.name for f in self.prompts_dir.glob("*.md")]

    def query(
        self,
        user_input: str,
        system_prompt: str | None = None,
        prompt_file: str | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None
    ) -> LLMResponse:
        """
        Send query to LLM provider.

        Args:
            user_input: User message content
            system_prompt: Optional system prompt text
            prompt_file: Optional prompt file to load from prompts directory
            model: Optional model override
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Returns:
            LLMResponse with content and metadata
        """
        try:
            # Build messages
            messages = []

            # Add system prompt
            if prompt_file:
                system_prompt = self.load_prompt(prompt_file)

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            messages.append({"role": "user", "content": user_input})

            # Make API call
            time.time()
            response = self.client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Extract response data
            content = response.choices[0].message.content or ""
            usage = getattr(response, 'usage', None)

            # Extract detailed token information
            if usage:
                total_tokens = getattr(usage, 'total_tokens', None)
                input_tokens = getattr(usage, 'prompt_tokens', None)
                output_tokens = getattr(usage, 'completion_tokens', None)
            else:
                total_tokens = None
                input_tokens = None
                output_tokens = None

            # Calculate cost estimate (rough approximation)
            cost = self._estimate_cost(total_tokens) if total_tokens else None

            return LLMResponse(
                content=content,
                provider=self.config.name,
                model=model or self.model,
                tokens_used=total_tokens,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost
            )

        except Exception as e:
            return LLMResponse(
                content="",
                provider=self.config.name,
                model=model or self.model,
                input_tokens=None,
                output_tokens=None,
                error=str(e)
            )

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None
    ) -> LLMResponse:
        """
        Multi-turn chat conversation.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Optional model override
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Returns:
            LLMResponse with content and metadata
        """
        try:
            response = self.client.chat.completions.create(
                model=model or self.config.chat_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            content = response.choices[0].message.content or ""
            usage = getattr(response, 'usage', None)

            # Extract detailed token information
            if usage:
                total_tokens = getattr(usage, 'total_tokens', None)
                input_tokens = getattr(usage, 'prompt_tokens', None)
                output_tokens = getattr(usage, 'completion_tokens', None)
            else:
                total_tokens = None
                input_tokens = None
                output_tokens = None

            cost = self._estimate_cost(total_tokens) if total_tokens else None

            return LLMResponse(
                content=content,
                provider=self.config.name,
                model=model or self.config.chat_model,
                tokens_used=total_tokens,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost
            )

        except Exception as e:
            return LLMResponse(
                content="",
                provider=self.config.name,
                model=model or self.config.chat_model,
                input_tokens=None,
                output_tokens=None,
                error=str(e)
            )

    def _estimate_cost(self, tokens: int | None) -> float | None:
        """Rough cost estimation (adjust based on actual provider pricing)."""
        if not tokens:
            return None

        # Rough estimates per 1M tokens
        cost_per_million = {
            LLMProvider.OPENAI: 5.0,  # GPT-4o-mini
            LLMProvider.OPENROUTER: 3.0,  # Claude 3.5 Sonnet
            LLMProvider.DEEPINFRA: 0.27,  # Llama 3.1 8B
            LLMProvider.FIREWORKS: 0.20,  # Llama 3.1 8B
        }

        rate = cost_per_million.get(self.provider_enum, 1.0)
        return (tokens / 1_000_000) * rate

    def get_provider_info(self) -> dict[str, Any]:
        """Get current provider information."""
        return {
            "provider": self.config.name,
            "model": self.model,
            "base_url": self.config.base_url,
            "available": self._ping_provider()
        }

    @classmethod
    def auto_select_provider(cls, preferred_order: list[str] | None = None) -> 'Promptinator':
        """
        Auto-select the first available provider.

        Args:
            preferred_order: List of provider names in preference order

        Returns:
            Promptinator instance with working provider
        """
        order = preferred_order or ["openai", "openrouter", "deepinfra", "fireworks"]

        for provider_name in order:
            try:
                return cls(provider=provider_name)
            except (ValueError, ConnectionError):
                continue

        raise RuntimeError("No working LLM providers found")


# Convenience functions for analyze_dialogues.py integration
def create_llm_client(provider: str = "openai", model: str | None = None) -> Promptinator:
    """Create LLM client instance."""
    return Promptinator(provider=provider, model=model)


def analyze_dialogue_with_llm(
    group_data: dict[str, Any],
    messages_text: list[str],
    promptinator: Promptinator,
    prompt_file: str = "prompt_1.md"
) -> str:
    """
    Analyze dialogue using specified LLM provider and prompt.

    Args:
        group_data: Group metadata dict
        messages_text: List of formatted message strings
        promptinator: Promptinator instance
        prompt_file: Prompt file to use

    Returns:
        Analysis result string
    """
    try:
        # Load and format prompt
        prompt_template = promptinator.load_prompt(prompt_file)

        # Prepare variables for template formatting
        messages_joined = chr(10).join(messages_text)

        # Create a custom formatter that handles the template variables
        formatted_prompt = prompt_template.replace(
            "{group['chat_name']}", str(group_data.get('chat_name', 'Unknown'))
        ).replace(
            "{group['message_count']}", str(group_data.get('message_count', 0))
        ).replace(
            "{group['total_words']:,}", f"{group_data.get('total_words', 0):,}"
        ).replace(
            "{group['participants']}", str(group_data.get('participants', 0))
        ).replace(
            "{group['avg_words_per_message']:.1f}", f"{group_data.get('avg_words_per_message', 0):.1f}"
        ).replace(
            "{chr(10).join(messages_text)}", messages_joined
        )

        # Get system prompt for dialogue analysis
        system_prompt = "You analyze online conversations of friends. You think and provide response in the language of the conversation."

        # Query LLM
        response = promptinator.query(
            user_input=formatted_prompt,
            system_prompt=system_prompt,
            temperature=0.7
        )

        if response.error:
            return f"Error analyzing dialogue: {response.error}"

        # Add metadata to response
        metadata = ""
        if response.tokens_used:
            if response.input_tokens and response.output_tokens:
                token_info = f"Tokens: {response.input_tokens:,}/{response.output_tokens:,} (in/out)"
            else:
                token_info = f"Tokens: {response.tokens_used:,}"

            metadata = f"\n\n✅ Analysis complete | Provider: {response.provider} | Model: {response.model} | {token_info}"
            if response.cost:
                metadata += f" | Cost: ${response.cost:.4f}"

        return response.content + metadata

    except Exception as e:
        return f"Error in dialogue analysis: {e}"


if __name__ == "__main__":
    # Demo usage
    try:
        # Try to auto-select provider
        promptinator = Promptinator.auto_select_provider()
        print(f"Using provider: {promptinator.get_provider_info()}")

        # List available models
        models = promptinator.get_available_models()
        print(f"Available models: {models[:5]}...")  # Show first 5

        # List prompts
        prompts = promptinator.list_available_prompts()
        print(f"Available prompts: {prompts}")

        # Test query
        response = promptinator.query("Hello, how are you?")
        print(f"Response: {response.content[:100]}...")

    except Exception as e:
        print(f"Demo failed: {e}")
