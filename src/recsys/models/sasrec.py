from rectools.models import SASRecModel
from rectools.models.nn.item_net import IdEmbeddingsItemNet

def make_sasrec(n_factors=64, n_blocks=2, n_heads=4, dropout_rate=0.2, session_max_len=50,
                loss="softmax", n_negatives=1, lr=1e-3, batch_size=256, epochs=5, seed=42):
    """Новая модель SASRec с заданными гиперпараметрами."""
    import torch
    torch.manual_seed(seed)
    return SASRecModel(
        n_factors=n_factors,
        n_blocks=n_blocks,
        n_heads=n_heads,
        dropout_rate=dropout_rate,
        session_max_len=session_max_len,
        loss=loss,
        n_negatives=n_negatives,
        lr=lr,
        batch_size=batch_size,
        epochs=epochs,
        deterministic=True,
        item_net_block_types=(IdEmbeddingsItemNet,),
        verbose=1,
    )