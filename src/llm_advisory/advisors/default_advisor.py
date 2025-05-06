from llm_advisory import LLMAdvisor

ADVISOR_INSTRUCTIONS = """
You are a advisor and provide advice based on the provided data."""


class DefaultAdvisor(LLMAdvisor):
    """Default advisor"""

    advisor_instructions = ADVISOR_INSTRUCTIONS
