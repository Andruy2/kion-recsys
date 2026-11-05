import pandas as pd

def build_eval(train, val):
    """Тёплые пользователи val и правильные ответы для них."""
    val_users = val["user_id"].drop_duplicates()
    warm_users = val_users[val_users.isin(train["user_id"])]
    val_warm = val[val["user_id"].isin(warm_users)].reset_index(drop=True)
    return warm_users, val_warm

def recall_at_k(recs, truth, k=10, normalize=False):
    """Средний Recall@k по пользователям из truth.

    normalize=True: знаменатель min(k, число правильных ответов),
    чтобы у пользователей с > k ответами максимум был 1.0.
    """
    top_k = recs[recs["rank"] <= k]
    hits = top_k.merge(truth[["user_id", "item_id"]], on=["user_id", "item_id"], how="inner")

    hits_per_user = hits.groupby("user_id").size().rename("hits")
    relevant_per_user = truth.groupby("user_id").size().rename("relevant")

    per_user = pd.concat([relevant_per_user, hits_per_user], axis=1)
    per_user["hits"] = per_user["hits"].fillna(0)

    denominator = per_user["relevant"].clip(upper=k) if normalize else per_user["relevant"]
    per_user["recall"] = per_user["hits"] / denominator

    return per_user["recall"].mean()