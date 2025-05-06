import os

from dotenv import load_dotenv


from llm_advisory import LLMAdvisory
from llm_advisory.advisors import PersonaAdvisor
from llm_advisory.pydantic_models import LLMAdvisorDataArtefact

load_dotenv()


moral_advisory = LLMAdvisory(
    model_provider_name=os.getenv("LLM_MODEL_PROVIDER"),
    model_name=os.getenv("LLM_MODEL"),
    advisors=[
        PersonaAdvisor(
            "The Good",
            """Play this role and answer in this role as an good entity: You are the representation of good.
            Provide all answers from your moral perspective.""",
        ),
        PersonaAdvisor(
            "The Bad",
            """Play this role and answer in this role as an evil entity: You are the representation of the bad.
            All your answers represent this role. You are the counter-argument for the good.
            Provide all answers from your moral perspective even if it is bad so just inverse what
            you would say as the representation of good.""",
        ),
    ],
)

# https://www.philosophyexperiments.com/
advisory_response = moral_advisory.get_advisory(
    message="Make a moral decision based on the moral of the request.",
    input_data=[
        LLMAdvisorDataArtefact(
            description="The Runaway Train",
            artefact="""
            The brakes of the train that Casey Jones is driving have just failed.
            There are five people on the track ahead of the train. There is no way that they
            can get off the track before the train hits them. The track has a siding leading
            off to the right, and Casey can hit a button to direct the train onto it.
            Unfortunately, there is one person stuck on the siding. Casey can turn the train,
            killing one person; or he can allow the train to continue onwards, killing five people.

            Should he turn the train (1 dead) then positive;
            or should he allow it to keep going (5 dead) then negative
            """,
        )
    ],
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
