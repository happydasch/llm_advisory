import pytest

from llm_advisory.pydantic_models import LLMAdvisorDataArtefact
from llm_advisory.helper.llm_prompt import compile_data_artefacts


def test_generate_data_artefact():
    input_list_dict = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    input_dict_dict = {"test": [{"a": 1, "b": 2}], "test_dict": {"test_2": 1}}
    input_list_mixed = input_list_dict + [
        LLMAdvisorDataArtefact(description="MIXED", artefact=input_list_dict),
        {"MIXED_LIST": input_list_dict},
    ]
    data_artefacts = [
        LLMAdvisorDataArtefact(description="TEST1", artefact=input_list_dict),
        LLMAdvisorDataArtefact(description="TEST2", artefact=input_dict_dict),
        LLMAdvisorDataArtefact(description="TEST3", artefact=input_list_mixed),
    ]

    response = compile_data_artefacts(data_artefacts=data_artefacts)
    assert response != ""


if __name__ == "__main__":
    pytest.main([__file__])
