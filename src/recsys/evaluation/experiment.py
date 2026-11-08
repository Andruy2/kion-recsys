from recsys.evaluation.metrics import build_eval
from rectools.dataset import Dataset
from rectools.metrics import MAP, NDCG, Recall, calc_metrics
from recsys.data.preprocessing import to_rectools

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
    """Добавляет строку эксперимента: модель, фолд, параметры и метрики — каждое в своей колонке."""
    results.append({"model": model, "fold": fold, **params, **metrics})

def prepare_fold(train, val):
    warm_users, truth = build_eval(train, val)
    return {"train": train, "dataset": Dataset.construct(to_rectools(train)),
            "warm_users": warm_users, "truth": truth}

