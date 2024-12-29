import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop)


class CategoricalBinning(BaseEstimator, TransformerMixin):
    """
    Binning categorical variables into specified bins.
    """

    def __init__(self, column, bins, labels):
        self.column = column
        self.bins = bins
        self.labels = labels

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        bin_mapping = {cat: label for label, cats in self.bins.items() for cat in cats}

        X[self.column] = X[self.column].map(bin_mapping)

        return X


class NumericBinning(BaseEstimator, TransformerMixin):
    """
    Binning numeric variables into specified bins, overwrites original column by default.
    """

    def __init__(self, column, bins, labels, overwrite=True):
        self.column = column
        self.bins = bins
        self.labels = labels
        self.overwrite = overwrite

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if self.overwrite:
            col_name = self.column
        else:
            col_name = self.column + "_binned"

        X[col_name] = pd.cut(
            X[self.column], bins=self.bins, labels=self.labels, right=False
        )

        return X
