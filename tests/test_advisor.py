import pytest

from llm_advisory import LLMAdvisory
from llm_advisory.advisors import PersonaAdvisor, DefaultAdvisor


def test_default_advisor():
    advisory = LLMAdvisory(
        advisors=[DefaultAdvisor("Test Advisor")],
        model_provider_name="ollama",
        model_name="gemma3",
    )
    advisory_response = advisory.get_advisory("Test initial message")

    assert advisory_response is not None


def test_person_advisor():
    advisory = LLMAdvisory(
        advisors=[PersonaAdvisor("Test person", "Test person for using in pytest")],
        model_provider_name="ollama",
        model_name="gemma3",
    )
    advisory_response = advisory.get_advisory("Test initial message")

    assert advisory_response is not None


if __name__ == "__main__":
    pytest.main([__file__])
