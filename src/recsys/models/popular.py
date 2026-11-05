import pandas as pd

def recommend_popular(train, users, k=10, days=7, n_candidates=100):
    """Топ-k популярных айтемов за последние days дней train,
    без уже просмотренных пользователем."""
    # 1. популярность за последние days дней
    recent = train[train["last_watch_dt"] > train["last_watch_dt"].max() - pd.Timedelta(days=days)]
    top = (recent["item_id"].value_counts()
           .head(n_candidates)
           .rename("pop")
           .reset_index())
    top["pop_rank"] = range(len(top))
    # 2. каждому пользователю — все кандидаты (декартово произведение)
    recs = pd.DataFrame({"user_id": users}).merge(top[["item_id", "pop_rank"]], how="cross")

    # 3. убрать просмотренное в train (anti-join)
    seen = train[["user_id", "item_id"]]
    recs = recs.merge(seen, on=["user_id", "item_id"], how="left", indicator=True)
    recs = recs[recs["_merge"] == "left_only"].drop(columns="_merge")

    # 4. оставить первые k по популярности у каждого
    recs = recs.sort_values(["user_id", "pop_rank"])
    recs["rank"] = recs.groupby("user_id").cumcount() + 1
    return recs[recs["rank"] <= k][["user_id", "item_id", "rank"]].reset_index(drop=True)