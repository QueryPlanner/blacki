"""Regression tests for the strict asynchronous warning policy."""

import asyncio
import inspect

import conftest
import pytest

pytest_plugins = ["pytester"]


def test_pytest_bootstrap_event_loop_is_held_for_session() -> None:
    """The bootstrap loop must remain alive until pytest session teardown."""
    loop = conftest._PYTEST_BOOTSTRAP_EVENT_LOOP

    assert loop is not None
    assert not loop.is_closed()


def test_pytest_sessionfinish_closes_bootstrap_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Session teardown must close its loop and restore the previous loop."""
    session_loop = conftest._PYTEST_BOOTSTRAP_EVENT_LOOP
    bootstrap_loop = asyncio.new_event_loop()
    previous_loop = asyncio.new_event_loop()
    monkeypatch.setattr(conftest, "_PYTEST_BOOTSTRAP_EVENT_LOOP", bootstrap_loop)
    monkeypatch.setattr(conftest, "_PYTEST_PREVIOUS_EVENT_LOOP", previous_loop)

    try:
        conftest.pytest_sessionfinish()

        assert bootstrap_loop.is_closed()
        assert not previous_loop.is_closed()
        assert asyncio.get_event_loop() is previous_loop
        assert conftest._PYTEST_BOOTSTRAP_EVENT_LOOP is None
        assert conftest._PYTEST_PREVIOUS_EVENT_LOOP is None
    finally:
        previous_loop.close()
        asyncio.set_event_loop(session_loop)


def test_async_pytest_run_closes_bootstrap_loop_without_warnings(
    pytester: pytest.Pytester,
) -> None:
    """An isolated async run must exit cleanly under strict warning filters."""
    lifecycle_hooks = "\n\n".join(
        inspect.getsource(hook)
        for hook in (
            conftest.pytest_sessionstart,
            conftest.pytest_sessionfinish,
        )
    )
    pytester.makeconftest(
        "\n".join(
            (
                "import asyncio",
                "import warnings",
                "",
                "import pytest",
                "",
                "_PYTEST_BOOTSTRAP_EVENT_LOOP = None",
                "_PYTEST_PREVIOUS_EVENT_LOOP = None",
                "",
                lifecycle_hooks,
            )
        )
    )
    pytester.makepyfile(
        """
        import asyncio
        import gc

        import pytest

        @pytest.mark.parametrize("iteration", range(8))
        @pytest.mark.asyncio
        async def test_async_runner(iteration):
            await asyncio.sleep(0)
            assert iteration >= 0

        def test_collect_discarded_loops():
            asyncio.set_event_loop(None)
            gc.collect()
        """
    )

    result = pytester.runpytest_subprocess(
        "--override-ini",
        "asyncio_default_fixture_loop_scope=function",
        "-W",
        "error::ResourceWarning",
        "-W",
        "error::pytest.PytestUnraisableExceptionWarning",
    )

    assert result.ret == pytest.ExitCode.OK
    result.assert_outcomes(passed=9)


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
