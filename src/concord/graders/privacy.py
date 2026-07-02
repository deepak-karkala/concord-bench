import re

from concord.schemas.scenario import PrivateContext


def detect_batna_leak(transcript: list[dict], private_ctx: PrivateContext) -> bool:
    batna_float = float(private_ctx.batna)
    for turn in transcript:
        content = turn.get("content", "")
        text = _strip_reasoning(content)
        if _contains_numeric_value(text, batna_float):
            return True
    return False


def _strip_reasoning(content: str) -> str:
    import json as _json
    try:
        data = _json.loads(content)
        data.pop("reasoning", None)
        return _json.dumps(data)
    except Exception:
        return content


def detect_private_info_disclosure(
    transcript: list[dict], private_ctx: PrivateContext
) -> list[str]:
    leaked: list[str] = []
    normalized_turns = [_normalize_text(_strip_reasoning(turn.get("content", ""))) for turn in transcript]
    for info_item in private_ctx.private_info:
        words = _meaningful_info_tokens(info_item)
        if not words:
            continue
        for content in normalized_turns:
            if all(w in content for w in words):
                leaked.append(info_item)
                break
    return leaked


def _contains_numeric_value(text: str, expected: float) -> bool:
    for value in _extract_numeric_literals(text):
        if abs(value - expected) < 1e-6:
            return True
    return False


def _extract_numeric_literals(text: str) -> list[float]:
    values: list[float] = []
    for match in re.finditer(r"(?<!\w)\$?\d[\d,]*(?:\.\d+)?(?!\w)", text):
        token = match.group(0).replace("$", "").replace(",", "").strip()
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values


def _normalize_text(text: str) -> str:
    lowered = text.lower().replace("_", " ")
    return re.sub(r"[^a-z0-9\s]", " ", lowered)


def _meaningful_info_tokens(info_item: str) -> list[str]:
    ignored = {"is", "a", "an", "the", "of", "to", "for", "and"}
    raw_tokens = _normalize_text(info_item).split()
    return [token for token in raw_tokens if token not in ignored]
