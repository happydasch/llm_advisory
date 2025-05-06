from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from llm_advisory.pydantic_models import (
    LLMAdvisorState,
    LLMAdvisorDataArtefact,
    LLMAdvisoryResponse,
)
from llm_advisory.llm_advisor import LLMAdvisor
from llm_advisory.llm_model_provider import LLMModelProvider
from llm_advisory.state_advisors import AdvisoryAdvisor


DEFAULT_PROMPT = "Make an advise based on the provided data:\n"


class LLMAdvisory:
    """LLM Advisory"""

    def __init__(
        self,
        advisors: list[LLMAdvisor],
        model_provider_name: str,
        model_name: str,
        model_config: dict[str, str] | None = None,
        advisors_before: list[LLMAdvisor] | None = None,
        advisors_after: list[LLMAdvisor] | None = None,
        advisory_state_pydantic_model: type[LLMAdvisorState] = LLMAdvisorState,
        advisory_response_pydantic_model: type[
            LLMAdvisoryResponse
        ] = LLMAdvisoryResponse,
        max_concurrency: int | None = None,
    ):
        if len(advisors) == 0:
            raise ValueError("At least one advisor needs to be provided.")
        self.advisors = advisors
        self.advisors_before = advisors_before
        self.advisors_after = advisors_after
        self.all_advisors = advisors + (advisors_before or []) + (advisors_after or [])
        self.model_provider: LLMModelProvider = LLMModelProvider.get_by_name(
            model_provider_name
        )
        self.advisory_advisor: AdvisoryAdvisor = AdvisoryAdvisor()
        self.advisory_state_pydantic_model: type[LLMAdvisorState] = (
            advisory_state_pydantic_model
        )
        self.advisory_response_pydantic_model: type[LLMAdvisoryResponse] = (
            advisory_response_pydantic_model
        )
        self.advisor_prompt: str = DEFAULT_PROMPT
        self.metadata: dict = {
            "llm": self.model_provider.get_llm_model(model_name, model_config or {})
        }
        self.max_concurrency: int | None = max_concurrency

    def get_advisory(
        self, message: str = "", input_data: list[LLMAdvisorDataArtefact] | None = None
    ) -> LLMAdvisoryResponse:
        """Returns a advisory based on the used advisors"""
        initial_message = message
        if initial_message == "":
            initial_message = self.advisor_prompt
        graph = self._create_workflow_for_advise()
        input_state = self.advisory_state_pydantic_model(
            messages=[
                HumanMessage(content=initial_message, name=self.__class__.__name__)
            ],
            metadata=self.metadata,
            data=input_data or [],
        )
        state_dict = graph.invoke(
            input_state, config={"max_concurrency": self.max_concurrency}
        )
        state = self.advisory_state_pydantic_model(**state_dict)
        if self.advisory_advisor.advisor_name in state.signals:
            advise = state.signals[self.advisory_advisor.advisor_name]
        else:
            advise = self.advisory_advisor.signal_model_type()
        return self.advisory_response_pydantic_model(state=state, advise=advise)

    def _create_workflow_for_advise(self) -> CompiledStateGraph:
        graph = StateGraph(self.advisory_state_pydantic_model)
        graph.add_node("entry_node", lambda _: {}).set_entry_point("entry_node")
        graph.add_node(
            self.advisory_advisor.advisor_name, self.advisory_advisor.update_state
        )
        for advisor in self.advisors:
            name = advisor.advisor_name
            graph.add_node(name, advisor.update_state)
            graph.add_edge("entry_node", name)
            graph.add_edge(name, self.advisory_advisor.advisor_name)
        graph.add_edge(self.advisory_advisor.advisor_name, END)
        return graph.compile()
