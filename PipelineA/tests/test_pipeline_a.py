"""Tests for Pipeline A using a stub LLM client."""

from PipelineA.pipeline_a import run_pipeline_a
from common.llm_client import StubLLMClient

FIXED_NARRATIVE = (
    "Type 2 Diabetes Mellitus -> TCF7L2 -> APP -> Alzheimer's Disease. "
    "TCF7L2 activates APP processing, which inhibits normal amyloid clearance."
)


# The result carries the model's reply as the narrative.
def test_run_pipeline_a_returns_the_narrative():
    client = StubLLMClient(fixed_response=FIXED_NARRATIVE)
    result = run_pipeline_a("Type 2 Diabetes Mellitus", "Alzheimer's Disease", client=client)

    assert result.disease1 == "Type 2 Diabetes Mellitus"
    assert result.disease2 == "Alzheimer's Disease"
    assert result.narrative == FIXED_NARRATIVE


# Both disease names appear in the prompt sent to the model.
def test_run_pipeline_a_prompts_with_both_disease_names():
    client = StubLLMClient(fixed_response=FIXED_NARRATIVE)
    run_pipeline_a("Type 2 Diabetes Mellitus", "Coronary Artery Disease", client=client)

    assert "Type 2 Diabetes Mellitus" in client.last_prompt
    assert "Coronary Artery Disease" in client.last_prompt
    assert "Do not say you need to look anything up" in client.last_prompt


# The result converts to a dict that json.dumps accepts.
def test_run_pipeline_a_to_dict_is_json_serializable():
    import json
    client = StubLLMClient(fixed_response=FIXED_NARRATIVE)
    result = run_pipeline_a("Type 2 Diabetes Mellitus", "Alzheimer's Disease", client=client)
    json.dumps(result.to_dict())  # Raises if the dict is not serializable.
