from implicit.als import AlternatingLeastSquares
from rectools.models import ImplicitALSWrapperModel


def make_als(factors=64, regularization=0.01, alpha=1.0, iterations=15, random_state=42):
    """Новая модель iALS с заданными гиперпараметрами (на CPU)."""
    return ImplicitALSWrapperModel(AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        alpha=alpha,
        iterations=iterations,
        random_state=random_state,
        use_gpu=False,
    ))