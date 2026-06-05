import argparse

from dms.cli import _explicit_model_overrides


def test_explicit_model_overrides_only_include_user_supplied_flags() -> None:
    args = argparse.Namespace(
        max_tokens=1200,
        temperature=0.0,
        timeout_seconds=120,
        _raw_argv=("run-temporal-extraction", "script.json", "--max-tokens", "1200"),
    )

    assert _explicit_model_overrides(args) == {"max_tokens": 1200}


def test_explicit_model_overrides_can_override_timeout_and_temperature() -> None:
    args = argparse.Namespace(
        max_tokens=2048,
        temperature=0.2,
        timeout_seconds=300,
        _raw_argv=("--temperature", "0.2", "--timeout-seconds", "300"),
    )

    assert _explicit_model_overrides(args) == {"temperature": 0.2, "timeout_seconds": 300}
