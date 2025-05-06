from llm_advisory.pydantic_models import (
    LLMAdvisorDataArtefact,
    LLMAdvisorDataArtefactOutputMode,
)
from llm_advisory.helper.llm_prompt import compile_data_artefacts

input_list_dict = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
input_dict_dict = {"test": [{"c": 5, "d": 6}, {"c": 2}], "test_dict": {"test_2": 7}}
input_list_mixed = input_list_dict + [
    LLMAdvisorDataArtefact(
        description="MIXED",
        artefact=input_list_dict,
        output_mode=LLMAdvisorDataArtefactOutputMode.MARKDOWN_TABLE,
    ),
    {"MIXED_LIST": input_list_dict},
]
input_dict_mixed = {
    "test_a": input_list_dict,
    "test_b": input_dict_dict,
    "test_c": input_list_mixed,
}
data_artefacts = [
    LLMAdvisorDataArtefact(
        description="TEST1",
        artefact=input_list_dict,
        output_mode=LLMAdvisorDataArtefactOutputMode.MARKDOWN_TABLE,
    ),
    LLMAdvisorDataArtefact(
        description="TEST2",
        artefact=input_dict_dict,
        output_mode=LLMAdvisorDataArtefactOutputMode.MARKDOWN_TABLE,
    ),
    LLMAdvisorDataArtefact(
        description="TEST3",
        artefact=input_list_mixed,
        output_mode=LLMAdvisorDataArtefactOutputMode.JSON_OBJECT,
    ),
    LLMAdvisorDataArtefact(
        description="TEST4",
        artefact=input_dict_mixed,
    ),
]

response = compile_data_artefacts(data_artefacts=data_artefacts)
assert response != ""

print(response)
