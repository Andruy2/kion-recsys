import pandas as pd


def preprocess_interactions(interactions, min_dur=300, max_dur=None):
    """Удаляет аномалии по total_dur и оставляет позитивные взаимодействия."""
    if max_dur is None:
        period = (interactions["last_watch_dt"].max()
                  - interactions["last_watch_dt"].min()
                  + pd.Timedelta(days=1))
        max_dur = period.total_seconds()

    mask = (interactions["total_dur"] >= min_dur) & (interactions["total_dur"] <= max_dur)
    return interactions[mask].reset_index(drop=True)


def time_split(df, n_days=7):
    """Делит df на (до последних n_days дней, последние n_days дней)."""
    split_dt = df["last_watch_dt"].max() - pd.Timedelta(days=n_days)
    before = df[df["last_watch_dt"] <= split_dt].reset_index(drop=True)
    after = df[df["last_watch_dt"] > split_dt].reset_index(drop=True)
    return before, after