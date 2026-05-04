_AWM_AVAILABLE: bool | None = None


def _check_awm() -> None:
    global _AWM_AVAILABLE
    if _AWM_AVAILABLE is not None:
        return
    try:
        import awm  # noqa: F401
        _AWM_AVAILABLE = True
    except ModuleNotFoundError:
        _AWM_AVAILABLE = False
        raise ModuleNotFoundError(
            "Concord scenario generation requires the [synth] extra. "
            "Install with: pip install concord-bench[synth]"
        )
