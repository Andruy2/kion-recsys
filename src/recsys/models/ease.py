from rectools.models import EASEModel


def make_ease(regularization=500.0):
    """Новая модель EASE с заданной регуляризацией."""
    return EASEModel(regularization=regularization)