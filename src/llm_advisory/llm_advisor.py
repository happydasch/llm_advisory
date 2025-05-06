from logging import getLogger
from typing import TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    ChatPromptTemplate,
)

from llm_advisory.pydantic_models import (
    LLMAdvisorState,
    LLMAdvisorSignal,
    LLMAdvisorMessagesInput,
    LLMAdvisorUpdateStateData,
)
from llm_advisory.helper.llm_prompt import (
    generate_description_from_pydantic_model,
    compile_data_artefacts,
)

T = TypeVar("T", bound=LLMAdvisorSignal)

logger = getLogger(__name__)


class LLMAdvisor:
    """LLM Advisor generic implementation"""

    # advisor instructions are used for the system prompt
    advisor_instructions = ""
    # default advisor prompt if no prompt message provided
    advisor_prompt = ""

    # the model to use for signal generation (LLMAdvisorSignal based)
    signal_model_type: type[LLMAdvisorSignal] = LLMAdvisorSignal
    # the model to use for state (LLMAdvisorState based)
    state_model_type: type[LLMAdvisorState] = LLMAdvisorState

    def __init__(
        self,
        messages_input_type: type[LLMAdvisorMessagesInput] = LLMAdvisorMessagesInput,
    ):
        # set the advisor settings
        self.advisor_name: str = self.__class__.__name__
        self.advisor_messages_input: LLMAdvisorMessagesInput = messages_input_type()
        self.advisor_system_prompt: str = (
            self.advisor_messages_input.get_system_prompt_template()
        )
        self.advisor_human_prompt: str = (
            self.advisor_messages_input.get_human_prompt_template()
        )
        # message input controls state update values to use
        # by setting a value, a system prompt will be added
        self.advisor_messages_input.advisor_instructions = self.advisor_instructions
        # prepare the advisor prompt, to unset the output, set to "" in update_state
        self.advisor_messages_input.advisor_prompt = self.advisor_prompt

    def __repr__(self):
        return f"{self.advisor_name} {type(self)}"

    def update_state(
        self, state: LLMAdvisorUpdateStateData
    ) -> LLMAdvisorUpdateStateData:
        """Default callback method for invoke"""
        self.advisor_messages_input.advisor_prompt = state.messages[0].content
        self.advisor_messages_input.advisor_data = compile_data_artefacts(state.data)
        self._update_state(state=state)

    def _update_state(
        self, state: LLMAdvisorUpdateStateData
    ) -> LLMAdvisorUpdateStateData:
        """Internal update state method

        messages_input:
        - Defaults are set in init, in update_state only changeable data
        - Data needs to be already present, no data will be generated
        - A human prompt will be set if an advisor_prompt is present
        - A description of the returning pydantic model will be always returned
        """
        # prepare messages input values for prompts
        messages_input = self.advisor_messages_input
        messages_input.advisor_signal_json = generate_description_from_pydantic_model(
            self.signal_model_type
        )
        messages_templates = []
        # system prompt
        if self.advisor_messages_input.advisor_instructions:
            messages_templates.append(
                SystemMessagePromptTemplate.from_template(
                    template=self.advisor_system_prompt, name=self.advisor_name
                )
            )
        # human prompt
        if messages_input.advisor_prompt or messages_input.advisor_data:
            messages_templates.append(
                HumanMessagePromptTemplate.from_template(
                    template=self.advisor_human_prompt, name=self.advisor_name
                )
            )
        template = ChatPromptTemplate.from_messages(messages_templates)
        messages = template.invoke(messages_input.model_dump()).to_messages()
        for message in messages:
            message.name = self.advisor_name
        # generated signal
        signal = self._generate_signal(
            state=state, messages=messages, pydantic_model=self.signal_model_type
        )
        advisor_message = AIMessage(
            content=signal.model_dump_json(indent=2),
            name=self.advisor_name,
        )
        return {
            "messages": [advisor_message],
            "signals": {self.advisor_name: signal},
            "conversations": {self.advisor_name: ([*messages, advisor_message])},
        }

    def _generate_signal(
        self,
        state: LLMAdvisorState,
        messages: list[BaseMessage],
        pydantic_model: type[T],
    ) -> T:
        try:
            signal = self._invoke_llm_model(
                state=state,
                messages=messages,
                pydantic_model=pydantic_model,
            )
        except ValueError as e:
            logger.error("Error generating signal: %s", e)
            signal = pydantic_model(
                advise="none",
                confidence=0,
                reasoning=f"Error generating: {e}",
            )
        return signal

    def _invoke_llm_model(
        self,
        state: LLMAdvisorState,
        messages: list[BaseMessage],
        pydantic_model: type[T],
    ) -> T:
        llm: BaseChatModel = state.metadata.get("llm")
        if llm is None:
            raise ValueError("llm not found in state metadata")
        structured_llm = llm.with_structured_output(
            schema=pydantic_model,
            include_raw=True,
            method="json_mode",
        )
        result = structured_llm.invoke(messages)
        if result["parsed"] is None:
            logger.error("%s no signal generated", self.advisor_name)
            raise ValueError(result)
        return result["parsed"]
