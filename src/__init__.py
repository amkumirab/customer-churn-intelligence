"""Customer churn intelligence package."""

from .data import FEATURE_COLUMNS, generate_customers, load_kaggle_telco
from .model import train_and_evaluate

__all__ = [
    "FEATURE_COLUMNS",
    "generate_customers",
    "load_kaggle_telco",
    "train_and_evaluate",
]
