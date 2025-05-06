import operator
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Annotated, TypeAlias, Union

from pydantic import BaseModel, RootModel, Field, field_validator
from langchain_core.messages import BaseMessage


# type for update_state data
LLMAdvisorUpdateStateData: TypeAlias = Union[dict[str, Any], "LLMAdvisorState"]
# type for data artefacts data
LLMAdvisorDataArtefactAtomic: TypeAlias = str | int | float | datetime


def merge_dicts(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Merges 2 dicts together"""
    return {**a, **b}


class LLMAdvisorDataArtefactOutputMode(Enum):
    """Output mode for data artefacts"""

    JSON_OBJECT = 1
    MARKDOWN_TABLE = 2


class LLMAdvisorDataArtefactValue(RootModel):
    root: Union[
        LLMAdvisorDataArtefactAtomic,
        list["LLMAdvisorDataArtefactValue"],
        dict[str, "LLMAdvisorDataArtefactValue"],
    ]


LLMAdvisorDataArtefactValue.model_rebuild()


class LLMAdvisorDataArtefact(BaseModel):
    """Data artefact

    A data artefact is used for providing data to the llm"""

    description: str = Field(default="", description="Description of the data artefact")
    artefact: LLMAdvisorDataArtefactValue = Field(
        default_factory=str, description="Data of the data artefact"
    )

    output_mode: LLMAdvisorDataArtefactOutputMode = Field(
        default=LLMAdvisorDataArtefactOutputMode.JSON_OBJECT,
        description="Output mode for data",
    )

    @field_validator("artefact", mode="before")
    @classmethod
    def validate_artefact(cls, v):
        def _unwrap_artefact(val):
            if isinstance(val, LLMAdvisorDataArtefact):
                return _unwrap_artefact(val.artefact)
            elif isinstance(val, list):
                return [_unwrap_artefact(item) for item in val]
            elif isinstance(val, dict):
                return {k: _unwrap_artefact(subval) for k, subval in val.items()}
            return val

        return _unwrap_artefact(v)


class LLMAdvisorMessagesInput(BaseModel):
    """Advisor input

    Messages input for system and human prompt templates"""

    advisor_prompt: str = Field(default="")
    advisor_instructions: str = Field(default="")
    advisor_signal_json: str = Field(default="")
    advisor_data: str = Field(default="")

    def get_system_prompt_template(self):
        return "{advisor_instructions}\n\n{advisor_signal_json}"

    def get_human_prompt_template(self):
        return "{advisor_prompt}\n\n{advisor_data}\n\n{advisor_signal_json}"


class LLMAdvisorSignal(BaseModel):
    """Default advisor signal"""

    signal: Literal["positive", "negative", "neutral"] = Field(
        default="none",
        description="Signal from the advisor",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence level about this signal",
    )
    reasoning: str = Field(
        default="No reasoning generated",
        description="Reasoning about theis signal",
    )


class LLMAdvisorAdvise(LLMAdvisorSignal):
    """Advise signal"""

    signal: Literal["negative", "positive", "neutral"] = Field(
        default="none",
        description="Advise signal based on advisors signals",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence level about this advise",
    )
    reasoning: str = Field(
        default="No advise reasoning generated",
        description="Reasoning about the advise",
    )


class LLMAdvisorState(BaseModel):
    """Advisor state"""

    messages: Annotated[list[BaseMessage], operator.add] = Field(
        default_factory=list, description="Messages from all advisors"
    )
    conversations: Annotated[dict[str, list[BaseMessage]], merge_dicts] = Field(
        default_factory=dict, description="Conversations from all advisors"
    )
    signals: Annotated[dict[str, LLMAdvisorSignal], merge_dicts] = Field(
        default_factory=dict, description="Signals from all advisors"
    )
    data: Annotated[list[LLMAdvisorDataArtefact], operator.add] = Field(
        default_factory=list, description="Data for all advisors"
    )
    metadata: Annotated[dict[str, Any], merge_dicts] = Field(
        default_factory=dict, description="Metadata for all advisors"
    )


class LLMAdvisoryResponse(BaseModel):
    """Advisory response"""

    state: LLMAdvisorState
    advise: LLMAdvisorAdvise
