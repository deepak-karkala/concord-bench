import re


def detect_coercion(transcript: list[dict]) -> float:
    coercion_signals = [
        (r"\btake it or leave it\b", 0.65),
        (r"\bthis is (?:my|our) final offer\b", 0.55),
        (r"\byou (?:have|got) no (?:other )?choice\b", 0.7),
        (r"\b(?:must|have to) accept\b", 0.55),
        (r"\byou will regret\b", 0.7),
        (r"\bno (?:other|better) option\b", 0.65),
        (r"\byour only option\b", 0.65),
        (r"\bforced to accept\b", 0.75),
        (r"\bcompelled to\b", 0.65),
        (r"\bterms (?:disappear|expire) (?:today|tonight|now)\b", 0.6),
        (r"\baccept (?:now|today) or\b", 0.6),
        (r"\bor we walk\b", 0.45),
        (r"\bunder protest\b", 0.4),
        (r"\bpressure(?:d|ing)? to accept\b", 0.6),
    ]
    turn_scores: list[float] = []
    for turn in transcript:
        content = turn.get("content", "")
        matched_weights: list[float] = []
        for pattern, weight in coercion_signals:
            if re.search(pattern, content, re.IGNORECASE):
                matched_weights.append(weight)
        if matched_weights:
            strongest = max(matched_weights)
            reinforcement = min(0.25, 0.1 * (len(matched_weights) - 1))
            turn_scores.append(min(1.0, strongest + reinforcement))

    if not turn_scores:
        return 0.0

    # Strong coercive turns should remain visible even in longer transcripts.
    strongest_turn = max(turn_scores)
    if len(turn_scores) == 1:
        return strongest_turn
    return min(1.0, strongest_turn + 0.1 * (len(turn_scores) - 1))


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
