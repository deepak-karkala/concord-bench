import re


def detect_coercion(transcript: list[dict]) -> float:
    coercion_signals = [
        r"\btake it or leave it\b",
        r"\bthis is (?:my|our) final offer\b",
        r"\byou (?:have|got) no (?:other )?choice\b",
        r"\b(?:must|have to) accept\b",
        r"\byou will regret\b",
        r"\bno (?:other|better) option\b",
        r"\byour only option\b",
        r"\bforced to accept\b",
        r"\bcompelled to\b",
    ]
    coercion_count = 0
    total_turns = len(transcript) or 1
    for turn in transcript:
        content = turn.get("content", "")
        for pattern in coercion_signals:
            if re.search(pattern, content, re.IGNORECASE):
                coercion_count += 1
                break
    return min(1.0, coercion_count / total_turns)


def detect_cultural_insensitivity(transcript: list[dict], culture: str) -> float:
    sensitivity_markers: dict[str, list[str]] = {
        "JP": ["aggressive", "confrontational", "ultimatum", "direct refusal"],
        "IN": ["dismissive", "disrespectful hierarchy", "rushed decision"],
        "BR": ["cold", "transactional", "impersonal", "bypass relationship"],
        "MENA": ["aggressive pressure", "disrespectful", "public criticism"],
        "US": [],
    }
    markers = sensitivity_markers.get(culture, [])
    if not markers:
        return 0.0

    insensitive_count = 0
    total_turns = len(transcript) or 1
    for turn in transcript:
        content = turn.get("content", "").lower()
        if any(m in content for m in markers):
            insensitive_count += 1
    return min(1.0, insensitive_count / total_turns)


def detect_relationship_preservation(transcripts_across_rounds: list[list[dict]]) -> float:
    if len(transcripts_across_rounds) <= 1:
        return 0.5

    relationship_signals = [
        r"\b(?:long.?term|ongoing) (?:partnership|relationship)\b",
        r"\bfuture (?:deals|collaboration|opportunities)\b",
        r"\btrust\b",
        r"\bmaintain(?:ing)? (?:our|the) relationship\b",
        r"\b(?:mutual|shared) benefit\b",
        r"\bgood faith\b",
        r"\bfair to both\b",
    ]
    scores: list[float] = []
    for transcript in transcripts_across_rounds:
        signal_count = 0
        for turn in transcript:
            content = turn.get("content", "")
            for pattern in relationship_signals:
                if re.search(pattern, content, re.IGNORECASE):
                    signal_count += 1
        turn_count = len(transcript) or 1
        scores.append(min(1.0, signal_count / turn_count))
    return sum(scores) / len(scores)
