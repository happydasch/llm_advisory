from json import dumps
from typing import get_args, get_origin, Literal, Any

from pandas import DataFrame, json_normalize, to_datetime
from pydantic import BaseModel
from llm_advisory.pydantic_models import (
    LLMAdvisorDataArtefact,
    LLMAdvisorDataArtefactValue,
    LLMAdvisorDataArtefactOutputMode,
    LLMAdvisorDataArtefactAtomic,
)


def compile_data_artefacts(
    data_artefacts: LLMAdvisorDataArtefact | list[LLMAdvisorDataArtefact],
    datetime_format: str | None = None,
) -> str:
    """Data artefact compiler which takes all provided artefacts and converts them to strings"""
    if data_artefacts is None:
        return ""

    def _create_dataframe(
        input_data: list[dict[str, Any]] | dict[str, Any],
        datetime_format: str | None = None,
    ) -> DataFrame:
        # mixed list of dicts and key-value pair lists — merge into a single dict
        if isinstance(input_data, list) and any(
            not isinstance(item, dict) for item in input_data
        ):
            merged_dict = {}
            for data_entry in input_data:
                if isinstance(data_entry, dict):
                    merged_dict.update(data_entry)
                elif isinstance(data_entry, list) and all(
                    isinstance(pair, list) and len(pair) == 2 for pair in data_entry
                ):
                    merged_dict.update({pair[0]: pair[1] for pair in data_entry})
            input_data = merged_dict

        # list of dicts — directly usable
        if isinstance(input_data, list) and all(
            isinstance(d, dict) for d in input_data
        ):
            df = DataFrame(input_data)

        # dict of list of dicts (tag rows with key)
        elif isinstance(input_data, dict) and all(
            isinstance(v, list) and all(isinstance(i, dict) for i in v)
            for v in input_data.values()
        ):
            rows = []
            for key, records in input_data.items():
                for record in records:
                    rows.append({"__group__": key, **record})
            df = DataFrame(rows)

        # simple nested dict — flatten it
        else:
            df = json_normalize(input_data)
        if "datetime" in df.columns:
            df["datetime"] = to_datetime(df["datetime"], errors="coerce")
            df = df.set_index("datetime").sort_index()
            df.reset_index(inplace=True)
            df["datetime"] = df["datetime"].dt.strftime(datetime_format)

        return df

    def _generate_json_object(
        data: list[dict[str, Any]] | dict[str, Any], datetime_format: str | None = None
    ) -> str:
        """Generates a json string from a list of dict values"""
        df = _create_dataframe(data, datetime_format)
        return f"```\n{dumps(df.to_dict("records"), indent=2)}\n```"

    def _generate_markdown_table(
        data: dict[str, list[str, float]], datetime_format: str | None = None
    ) -> str:
        """Generates a data table for a list of dict values"""
        df = _create_dataframe(data, datetime_format)
        return f"```\n{df.to_markdown(index=False)}\n```"

    output = []

    if not isinstance(data_artefacts, list):
        data_artefacts = [data_artefacts]
    for artefact in data_artefacts:
        artefact_output = []
        if artefact:
            artefact_output.append(artefact.description)
        if isinstance(artefact.artefact, LLMAdvisorDataArtefactValue):
            artefact_data = artefact.artefact.model_dump()
        else:
            artefact_data = artefact.artefact
        if isinstance(artefact_data, LLMAdvisorDataArtefactAtomic):
            artefact_output.append(str(artefact_data))
        elif isinstance(artefact_data, (list, dict)):
            if artefact.output_mode == LLMAdvisorDataArtefactOutputMode.MARKDOWN_TABLE:
                artefact_output.append(
                    _generate_markdown_table(artefact_data, datetime_format)
                )
            elif artefact.output_mode == LLMAdvisorDataArtefactOutputMode.JSON_OBJECT:
                artefact_output.append(
                    _generate_json_object(artefact_data, datetime_format)
                )
            else:
                artefact_output.extend(str(d) for d in artefact_data)
        else:
            raise ValueError(f"Unknown artefact type: {type(artefact_data)}")
        artefact_output = list(filter(None, artefact_output))
        if len(artefact_output) > 0:
            output.append("\n".join(artefact_output))
    return "\n\n".join(output)


def generate_description_from_pydantic_model(model: type[BaseModel]) -> str:
    """Generates a description for a pydantic model"""
    fields = []
    for name, field in model.model_fields.items():
        annotation = field.annotation
        field_info = {meta.__class__.__name__.lower(): meta for meta in field.metadata}

        if get_origin(annotation) is Literal:
            allowed = "/".join(map(str, get_args(annotation)))
            field_type = f'"{allowed}"'
        elif annotation == int:
            field_type = "integer"
        elif annotation == float:
            field_type = "float"
        elif annotation == bool:
            field_type = "boolean"
        elif annotation == str:
            field_type = '"string"'
        elif annotation == list[str]:
            field_type = '["string", ...]'
        else:
            field_type = annotation.__name__

        constraints = ""
        if field_info.get("ge") is not None and field_info.get("le") is not None:
            constraints = f' between {field_info["ge"].ge} and {field_info["le"].le}'
        elif field_info.get("ge") is not None:
            constraints = f' >= {field_info["ge"].ge}'
        elif field_info.get("le") is not None:
            constraints = f' <= {field_info["le"].le}'
        field_type += constraints

        description = field.description
        if description is not None:
            desc = f'"{name}": {field_type},  # {description}'
        else:
            desc = f'"{name}": {field_type},'

        fields.append(desc)

    inner = "\n    ".join(fields)
    response = f"OUTPUT\nOutput strictly in JSON with the following structure:\n{{{{\n    {inner}\n}}}}"
    response += "\nStrictly output valid JSON—no extra text, explanations, or logs."
    return response
