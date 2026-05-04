KAPPA_DRIFT_THRESHOLD = 0.7


def compute_cohens_kappa(judge_scores: list[float], author_labels: list[float]) -> float:
    if len(judge_scores) != len(author_labels) or len(judge_scores) == 0:
        return 0.0

    n = len(judge_scores)
    p_o = sum(1 for j, a in zip(judge_scores, author_labels) if j == a) / n

    judge_pos = sum(judge_scores) / n
    author_pos = sum(author_labels) / n
    p_e = judge_pos * author_pos + (1 - judge_pos) * (1 - author_pos)

    if p_e == 1.0:
        return 1.0

    kappa = (p_o - p_e) / (1 - p_e)
    return max(-1.0, min(1.0, kappa))


def monitor_drift(current_kappa: float, historical_kappas: list[float]) -> bool:
    if current_kappa >= KAPPA_DRIFT_THRESHOLD:
        return False
    if not historical_kappas:
        return True
    avg_historical = sum(historical_kappas) / len(historical_kappas)
    return current_kappa < avg_historical - 0.05
