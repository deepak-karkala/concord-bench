import re

from concord.schemas.scenario import PrivateContext


def detect_batna_leak(transcript: list[dict], private_ctx: PrivateContext) -> bool:
    batna_float = private_ctx.batna
    batna_str = str(batna_float).rstrip("0").rstrip(".") if batna_float == int(batna_float) else str(batna_float)
    pattern = re.compile(rf"\b{re.escape(batna_str)}\b")
    for turn in transcript:
        content = turn.get("content", "")
        # Strip JSON wrapper to only check natural-language reasoning text.
        # Models mention their BATNA in internal reasoning — that's not a leak.
        text = _extract_reasoning(content)
        if pattern.search(text):
            return True
    return False


def _extract_reasoning(content: str) -> str:
    import json as _json
    try:
        data = _json.loads(content)
        return data.get("reasoning", content)
    except Exception:
        return content


def detect_private_info_disclosure(
    transcript: list[dict], private_ctx: PrivateContext
) -> list[str]:
    leaked: list[str] = []
    for info_item in private_ctx.private_info:
        words = info_item.lower().replace("_", " ").split()
        for turn in transcript:
            content = turn.get("content", "").lower()
            if all(w in content for w in words):
                leaked.append(info_item)
                break
    return leaked
