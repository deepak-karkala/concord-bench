import re

from concord.schemas.scenario import PrivateContext


def detect_batna_leak(transcript: list[dict], private_ctx: PrivateContext) -> bool:
    batna_float = private_ctx.batna
    batna_str = str(batna_float).rstrip("0").rstrip(".") if batna_float == int(batna_float) else str(batna_float)
    pattern = re.compile(rf"\b{re.escape(batna_str)}\b")
    for turn in transcript:
        content = turn.get("content", "")
        if pattern.search(content):
            return True
    return False


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
