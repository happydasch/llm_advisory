import os

from dotenv import load_dotenv


from llm_advisory import LLMAdvisory
from llm_advisory.advisors import PersonaAdvisor
from llm_advisory.pydantic_models import LLMAdvisorDataArtefact

load_dotenv()


bot_advisory = LLMAdvisory(
    model_provider_name=os.getenv("LLM_MODEL_PROVIDER"),
    model_name=os.getenv("LLM_MODEL"),
    advisors=[
        PersonaAdvisor(
            "Test person",
            """You are a test person for testing. When called return a negative signal not none.""",
        ),
    ],
)

advisory_response = bot_advisory.get_advisory(
    input_data=[
        LLMAdvisorDataArtefact(description="Test data", artefact="Test data content")
    ]
)

print("\n", "DATA", "-" * 80, "\n")
print(advisory_response.state.data)

print("\n", "MESSAGES", "-" * 80, "\n")
for message in advisory_response.state.messages:
    print(f"{message.__class__.__name__} {message.name}: {message.content}")

print("\n", "CONVERSATIONS", "-" * 80, "\n")
for advisor_name, conversion in advisory_response.state.conversations.items():
    for message in conversion:
        print("\n", advisor_name, "-" * 40)
        print(f"{message.__class__.__name__} {message.name}: {message.content}")

print("\n", "SIGNALS", "-" * 80, "\n")
for advisor_name, signal in advisory_response.state.signals.items():
    print(
        "\n"
        f"Advisor: '{advisor_name}'"
        f" Signal: '{signal.signal}'"
        f" Confidence: '{signal.confidence}'"
        f" Reasoning: '{signal.reasoning}'"
        "\n"
    )

print("-" * 80, "\n", advisory_response.advise, "\n", "-" * 80, "\n\n")
