from enum import Enum

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama


class LLMModelProvider(str, Enum):
    """Supported model provider"""

    OPENAI = "openai"
    OLLAMA = "ollama"

    @classmethod
    def get_by_name(cls, llm_provider_name: str) -> "LLMModelProvider":
        """Returns a model provider by name"""
        for provider in cls:
            if provider.value == llm_provider_name.lower():
                return provider
        raise ValueError(f"No matching model provider for '{llm_provider_name}'")

    def get_llm_model(self, model_config: dict[str, str]) -> BaseChatModel:
        """Returns the model with the given name from current model provider"""
        if self is LLMModelProvider.OPENAI:
            model_name = model_config.get("OPENAI_MODEL_NAME", "gpt-4o")
            api_key = model_config.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Please provide your OPENAI_API_KEY")
            return ChatOpenAI(model=model_name, api_key=api_key)
        elif self is LLMModelProvider.OLLAMA:
            model_name = model_config.get("OLLAMA_MODEL_NAME", "gemma3:12b")
            base_url = model_config.get("OLLAMA_BASE_URL", "http://localhost:11434")
            return ChatOllama(model=model_name, base_url=base_url)
        else:
            raise ValueError(f"Unsupported LLM Provider {self.__class__.__name__}")
