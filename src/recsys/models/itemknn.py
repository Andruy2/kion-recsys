from implicit.nearest_neighbours import BM25Recommender, CosineRecommender, TFIDFRecommender
from rectools.models import ImplicitItemKNNWrapperModel


def make_itemknn(similarity, K, B=None):
    """Новая модель ItemKNN с заданной мерой похожести."""
    if similarity == "cosine":
        return ImplicitItemKNNWrapperModel(CosineRecommender(K=K))
    if similarity == "tfidf":
        return ImplicitItemKNNWrapperModel(TFIDFRecommender(K=K))
    if similarity == "bm25":
        return ImplicitItemKNNWrapperModel(BM25Recommender(K=K, B=B))
    raise ValueError(f"Неизвестная мера похожести: {similarity}")