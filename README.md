# LLM Advisory

This framework allows to easily create multiple advisors using large language models. It uses a simple agent workflow which can be customized.

LLM Advisory uses `langchain` and `langgraph` for model and workflow handling, `pydantic` for data models. It is easily extendable and new advisors can be created by extending the core advisor. Support for `OpenAI` and `Ollama` is available.

The advisory object is created by `llm_advisory = LLMAdvisory(advisors=[DefaultAdvisor()])` with at least on advisor.

The advisors are invoked with a custom input prompt (or a default prompt to return a advisory) by calling `llm_advisory.get_advisory(...)`
which returns a pydantic model representing a signal containing the advisory.

**what can this be used for?**

`LLM Advisory` is a generic toolkit for topic based advisories. Each advisor takes the same raw data but focuses on its specialty.

By dividing the work, you:

1. Avoid confusion — each advisor has a narrow, well-defined role.
2. Get clearer signals — instead of one muddled answer, you get targeted, high-confidence advice from the “right” advisor.

Projects using `LLM Advisory`:

- [Backtrader LLM Advisory](https://github.com/happydasch/bt_llm_advisory) - A backtrader trading strategy advisory, which generates trade advises. Multiple advisors are available which generate data from a running strategy.

## Quickstart

Set up the LLMAdvisory with ChatGPT:

```python
from llm_advisory import LLMAdvisory
from llm_advisory.advisors import DefaultAdvisor
from llm_advisory.pydantic_models import LLMAdvisorDataArtefact

llm_advisory = LLMAdvisory(
    model_provider="openai",
    model_name="gpt-4o",
    model_config={"OPENAI_API_KEY": "xxx"},
    advisors=[DefaultAdvisor()]
)
```

Ask the advisory for a advise:

```python

advisory_response = llm_advisory.get_advisory(
    message="Is this a positive or negative message",
    data=[
        LLMAdvisorDataArtefact(
            description="Message",
            artefact="This is a positive message",
        )
    ]
)
```

Processing the advise:

```python
print(
    advisory_response.advise.signal,
    advisory_response.advise.confidence,
    advisory_response.advise.reasoning,
)
```

## Advisors

- `DefaultAdvisor`: Default advisor with no speciality
- `PersonaAdvisor`: Default advisor with additional personality specified

## State Advisors

- `AdvisoryAdvisor`: Advisor that creates the final advisory

## Examples

TODO add text how to create a basic advisory object

- Provide a advise from advisors based on provided data (moral)
- Create a advisor based on a famous trading Persona or a fictional Persona with a custom Persona description ()

## Frequently Asked Questions

- **My advisory is running slow when using ollama**
  Set `max_concurrency` to a lower value when creating the adivsory.

## Future functionality

A list with possible future functions.

- Flexible workflow
- Fetch additional data from provided sources, per advisor
- compile experience into memory for self improvement
- add observer for advisors to overwatch their behaviour
- add feedback for advises from trading so they can learn
