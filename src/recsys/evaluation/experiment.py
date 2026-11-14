import pandas as pd
import os
from recsys.evaluation.metrics import build_eval
from rectools.dataset import Dataset
from rectools.metrics import MAP, NDCG, Recall, calc_metrics
from recsys.data.preprocessing import shuffle_within_day, to_rectools

METRICS = {
    "recall@10": Recall(k=10),
    "ndcg@10": NDCG(k=10),
    "ndcg@10_ach": NDCG(k=10, divide_by_achievable=True),
    "map@10": MAP(k=10),
}


def evaluate(recs, truth):
    """Метрики рекомендаций против правильных ответов."""
    return calc_metrics(METRICS, reco=recs, interactions=truth[["user_id", "item_id"]])


def log_result(results, model, params, fold, metrics, path=None):
    """Добавляет строку в журнал и, если указан path, сразу дописывает её в CSV."""
    row = {"model": model, "fold": fold, **params, **metrics}
    results.append(row)
    if path is not None:
        pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)

def prepare_fold(train, val, shuffle_seed=None):
    """Всё, что нужно для оценки на одном фолде."""
    warm_users, truth = build_eval(train, val)
    train_for_model = train if shuffle_seed is None else shuffle_within_day(train, shuffle_seed)
    return {"train": train, "dataset": Dataset.construct(to_rectools(train_for_model)),
            "warm_users": warm_users, "truth": truth}

def run_model(results, name, params, make_model, fold, fold_name, k=10, path=None):
    """Обучает новую модель на фолде, считает метрики и пишет их в журнал."""
    model = make_model()
    model.fit(fold["dataset"])
    recs = model.recommend(users=fold["warm_users"], dataset=fold["dataset"], k=k, filter_viewed=True)
    log_result(results, name, params, fold_name, evaluate(recs, fold["truth"]), path)
    return recs