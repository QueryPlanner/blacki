"""Tests for deterministic prompt evaluation metrics."""

from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_metrics import EvalMetric, EvalStatus
from google.genai import types

from eval.prompt_metrics import concise_response_score


def _invocation(*parts: types.Part) -> Invocation:
    return Invocation(
        user_content=types.Content(
            role="user", parts=[types.Part.from_text(text="Question")]
        ),
        final_response=types.Content(role="model", parts=list(parts)),
    )


def test_concise_response_metric_passes_short_answer() -> None:
    invocation = _invocation(types.Part.from_text(text="A concise answer."))

    result = concise_response_score(
        EvalMetric(metric_name="concise_response_score"),
        [invocation],
        [invocation],
        None,
    )

    assert result.overall_score == 1.0
    assert result.overall_eval_status is EvalStatus.PASSED
    assert result.per_invocation_results[0].expected_invocation is invocation


def test_concise_response_metric_rejects_long_or_thought_text() -> None:
    long_invocation = _invocation(types.Part.from_text(text=" ".join(["word"] * 101)))
    thought_invocation = _invocation(
        types.Part(text="Hidden thought", thought=True),
        types.Part.from_text(text="Answer"),
    )

    result = concise_response_score(
        EvalMetric(metric_name="concise_response_score"),
        [long_invocation, thought_invocation],
        None,
        None,
    )

    assert result.overall_score == 0.0
    assert result.overall_eval_status is EvalStatus.FAILED
    assert all(
        invocation_result.eval_status is EvalStatus.FAILED
        for invocation_result in result.per_invocation_results
    )


def test_concise_response_metric_handles_no_invocations() -> None:
    result = concise_response_score(
        EvalMetric(metric_name="concise_response_score"), [], None, None
    )

    assert result.overall_score is None
    assert result.overall_eval_status is EvalStatus.NOT_EVALUATED
