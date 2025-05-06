from enum import Enum

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama


class LLMOpenAIModelNames(str, Enum):
    """Supported open ai model names"""

    GPT_4_5 = "gpt-4.5-preview"
    GPT_4_O = "gpt-4o"
    O3 = "o3"
    O4_MINI = "o4-mini"


class LLMOllamaModelNames(str, Enum):
    """Supported ollama model names"""

    GEMMA_3 = "gemma3"
    GEMMA_3_12B = "gemma3:12b"
    GEMMA_3_27B = "gemma3:27b"
    QWEN_3 = "qwen3"
    QWEN_2_5 = "qwen2.5"
    QWEN_2_5_32B = "qwen2.5:32b"
    LLAMA_3_1 = "llama3.1:latest"
    LLAMA_INSTRUCT_3_3_70B = "llama3.3:70b-instruct-q4_0"
    MISTRAL_SMALL_3_1 = "mistral-small3.1"
    DEEPSEEK_R1 = "deepseek-r1"


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

    def get_llm_model(
        self, model_name: str, model_config: dict[str, str] | None = None
    ) -> BaseChatModel:
        """Returns the model with the given name from current model provider"""
        llm_models = self.get_model_names_enum()
        if model_name not in llm_models:
            raise ValueError(
                f"Model {model_name} not supported by provider {self.__class__.__name__}"
            )
        if self is LLMModelProvider.OPENAI:
            api_key = model_config.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Please provide your OPENAI_API_KEY")
            return ChatOpenAI(model=model_name, api_key=api_key)
        elif self is LLMModelProvider.OLLAMA:
            base_url = model_config.get("OLLAMA_BASE_URL", "http://localhost:11434")
            return ChatOllama(model=model_name, base_url=base_url)
        else:
            raise ValueError(f"Unsupported LLM Provider {self.__class__.__name__}")

    def get_model_names_enum(self) -> LLMOpenAIModelNames | LLMOllamaModelNames:
        """Returns the model names enum for current model provider"""
        if self is LLMModelProvider.OPENAI:
            return LLMOpenAIModelNames
        elif self is LLMModelProvider.OLLAMA:
            return LLMOllamaModelNames
        else:
            raise ValueError(f"No model for provider {self.__class__.__name__} found")

    def get_model_names(self) -> list[str]:
        """Returns a list of model names for current model provider"""
        return [model_name.value for model_name in self.get_model_names_enum()]
