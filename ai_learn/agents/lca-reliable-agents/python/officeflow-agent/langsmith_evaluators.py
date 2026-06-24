"""LangSmith code evaluators for agent_v3_gemini.

Create local evaluators first and test with `evaluate(..., evaluators=[...])`.

Evaluator functions follow the `(run, example)` signature for offline/dataset use.
They are intentionally simple and avoid external dependencies so they can be uploaded
with the LangSmith CLI if desired.
"""
from typing import Any, Dict


def _get_run_outputs(run: Any) -> Dict[str, Any]:
    """Normalize run outputs for both local RunTree objects and uploaded dicts."""
    return run.outputs if hasattr(run, "outputs") else (run.get("outputs", {}) if isinstance(run, dict) else {})


def response_nonempty_evaluator(run, example):
    """Return score=1 when the agent returned a non-empty `output` string.

    Example dataset usage:
    - example.outputs may contain expected fields, but this evaluator only checks
      that the agent produced some textual response.
    """
    run_outputs = _get_run_outputs(run) or {}
    output = run_outputs.get("output") or run_outputs.get("text") or run_outputs.get("final")
    if output and str(output).strip():
        return {"score": 1, "comment": "Agent produced non-empty output."}
    return {"score": 0, "comment": "Empty or missing output field."}


def contains_expected_evaluator(run, example):
    """Return score=1 when the agent's output contains a substring specified in the example.

    Dataset authors should include `expected_contains` either at the top-level of the
    example (i.e. `example.get('expected_contains')`) or under `example.outputs`.

    This evaluator is useful for simple content checks, e.g. verifying that a
    product-related question mentions a product recommendation or that policy
    questions reference "returns" / "shipping" text.
    """
    run_outputs = _get_run_outputs(run) or {}
    example_outputs = example.outputs if hasattr(example, "outputs") else (example.get("outputs", {}) if isinstance(example, dict) else {})

    expected = example.get("expected_contains") or example_outputs.get("expected_contains")
    if not expected:
        return {"score": 0, "comment": "No `expected_contains` provided in example."}

    output = run_outputs.get("output") or run_outputs.get("text") or ""
    try:
        match = str(expected).lower() in str(output).lower()
    except Exception:
        match = False

    return {"score": 1 if match else 0, "comment": f"Expected substring present: {match}."}


__all__ = [
    "response_nonempty_evaluator",
    "contains_expected_evaluator",
]
