from llm_advisory.llm_advisor import LLMAdvisor


ADVISOR_INSTRUCTIONS = """
You are a advisor and provide advice based on the provided data.

Your advisory needs to include specialized knowledge, which you will also take into account
when creating the advise. You create the advise from the perspective of this personality:

{name}: {personality}"""


class PersonaAdvisor(LLMAdvisor):
    """Advisor with a persona"""

    advisor_instructions = ADVISOR_INSTRUCTIONS

    def __init__(
        self,
        # name of the persona, can also be a famous person
        person_name: str,
        # description of the personality, which also can contain informations about needed knowledge
        personality: str,
    ):
        super().__init__()
        self.advisor_name_default = self.advisor_name
        self.advisor_name = f"{self.advisor_name_default}{person_name.replace(" ", "")}"
        self.advisor_messages_input.advisor_instructions = (
            self.advisor_messages_input.advisor_instructions.format(
                name=person_name, personality=personality
            )
        )
        self.person_name = person_name
        self.personality = personality
