import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from src.config import INVALID_VALUES, TEXT_TO_NUMERIC_MAPPING

class GarbageCleaner(BaseEstimator, TransformerMixin):
    """
    Custom transformer to clean garbage values from specific columns.
    - Replaces 'Low'/'High' with numeric values.
    - Replaces invalid strings with NaN.
    - Coerces to numeric.
    """
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.columns is None:
            return X
            
        for col in self.columns:
            if col in X.columns:
                # Map text to numbers if applicable
                X[col] = X[col].replace(TEXT_TO_NUMERIC_MAPPING)
                
                # Replace invalid values with NaN
                X[col] = X[col].replace(INVALID_VALUES, np.nan)
                
                # Coerce to numeric
                X[col] = pd.to_numeric(X[col], errors='coerce')
        return X

class DataFrameSelector(BaseEstimator, TransformerMixin):
    """
    Selects specific columns from a DataFrame.
    """
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names]
