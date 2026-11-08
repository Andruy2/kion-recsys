from implicit.nearest_neighbours import CosineRecommender, TFIDFRecommender, BM25Recommender
from rectools.models import ImplicitItemKNNWrapperModel


def make_itemknn(similarity, K, B=None):
    if similarity == "cosine":
        return ImplicitItemKNNWrapperModel(CosineRecommender(K=K))
    if similarity == "tfidf":
        return ImplicitItemKNNWrapperModel(TFIDFRecommender(K=K))
    return ImplicitItemKNNWrapperModel(BM25Recommender(K=K, B=B))