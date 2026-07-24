"""Regression tests for the strict asynchronous warning policy."""

import pytest

pytest_plugins = ["pytester"]


def test_unawaited_coroutine_warning_fails_pytest(
    pytester: pytest.Pytester,
) -> None:
    """A leaked coroutine must make an isolated pytest run exit non-zero."""
    pytester.makepyfile(
        """
        import gc

        async def leaked_coroutine():
            return None

        def test_leaks_coroutine():
            leaked_coroutine()
            gc.collect()
        """
    )

    result = pytester.runpytest(
        "--override-ini",
        "asyncio_default_fixture_loop_scope=function",
        "-W",
        "error::RuntimeWarning",
        "-W",
        "error::pytest.PytestUnraisableExceptionWarning",
    )

    assert result.ret != pytest.ExitCode.OK
    result.assert_outcomes(failed=1)
