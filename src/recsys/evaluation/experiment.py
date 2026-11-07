from rectools.metrics import MAP, NDCG, Recall, calc_metrics

METRICS = {
    "recall@10": Recall(k=10),
    "ndcg@10": NDCG(k=10),
    "ndcg@10_ach": NDCG(k=10, divide_by_achievable=True),
    "map@10": MAP(k=10),
}


def evaluate(recs, truth):
    """Метрики рекомендаций против правильных ответов."""
    return calc_metrics(METRICS, reco=recs, interactions=truth[["user_id", "item_id"]])


def log_result(results, model, params, fold, metrics):
    """Добавляет строку эксперимента в список results."""
    results.append({"model": model, "params": str(params), "fold": fold, **metrics})