"""Deterministic metrics for the prompt behavior evaluation set."""

from __future__ import annotations

from typing import TYPE_CHECKING

from google.adk.evaluation.eval_metrics import EvalStatus
from google.adk.evaluation.evaluator import EvaluationResult, PerInvocationResult

if TYPE_CHECKING:
    from google.adk.evaluation.eval_case import ConversationScenario, Invocation
    from google.adk.evaluation.eval_metrics import EvalMetric


def concise_response_score(
    eval_metric: EvalMetric,
    actual_invocations: list[Invocation],
    expected_invocations: list[Invocation] | None,
    conversation_scenario: ConversationScenario | None,
) -> EvaluationResult:
    """Pass responses with at most 100 words and no exposed thought text."""
    del eval_metric, conversation_scenario
    per_invocation_results = []

    for index, actual in enumerate(actual_invocations):
        expected = (
            expected_invocations[index]
            if expected_invocations and index < len(expected_invocations)
            else None
        )
        parts = actual.final_response.parts if actual.final_response else []
        word_count = sum(len((part.text or "").split()) for part in parts or [])
        has_exposed_thought = any(
            bool(part.text and part.thought) for part in parts or []
        )
        score = float(word_count <= 100 and not has_exposed_thought)
        per_invocation_results.append(
            PerInvocationResult(
                actual_invocation=actual,
                expected_invocation=expected,
                score=score,
                eval_status=(EvalStatus.PASSED if score == 1.0 else EvalStatus.FAILED),
            )
        )

    if not per_invocation_results:
        return EvaluationResult()

    overall_score = sum(result.score or 0.0 for result in per_invocation_results) / len(
        per_invocation_results
    )
    return EvaluationResult(
        overall_score=overall_score,
        overall_eval_status=(
            EvalStatus.PASSED if overall_score == 1.0 else EvalStatus.FAILED
        ),
        per_invocation_results=per_invocation_results,
    )
