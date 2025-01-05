import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DropColumns(BaseEstimator, TransformerMixin):
    """
    Transformer to drop specified columns from a DataFrame.

    Args:
        columns_to_drop (list of str): List of column names to drop.

    Methods:
        fit(X, y=None):
            Fits the transformer. Returns self (no changes made during fit).

        transform(X):
            Drops the specified columns from the input DataFrame.

        set_output(transform='pandas'):
            Sets the desired output format for the transformer.

    Example:
        >>> transformer = DropColumns(columns_to_drop=['col1', 'col2'])
        >>> X_transformed = transformer.transform(X)
    """

    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop)

    def set_output(self, transform='pandas'):
        self.output = transform


class CategoricalBinning(BaseEstimator, TransformerMixin):
    """
    Transformer for binning categorical variables into specified groups.

    Args:
        column (str): Name of the column to bin.
        bins (dict): Dictionary mapping bin labels to lists of categories.
        labels (list of str): List of bin labels in order of appearance.
        overwrite (bool, optional): Whether to overwrite the original column. Defaults to True.

    Methods:
        fit(X, y=None):
            Fits the transformer. Returns self (no changes made during fit).

        transform(X):
            Bins the specified categorical column based on the provided mapping.

        set_output(transform='pandas'):
            Sets the desired output format for the transformer.

    Example:
        >>> binning = CategoricalBinning(
        ...     column='education',
        ...     bins={'Low': ['Primary', 'Middle'], 'High': ['High School', 'College']},
        ...     labels=['Low', 'High'],
        ... )
        >>> X_transformed = binning.transform(X)
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
            col_name = self.column + '_binned'

        bin_mapping = {cat: label for label, cats in self.bins.items() for cat in cats}

        X[col_name] = X[self.column].map(bin_mapping)

        return X

    def set_output(self, transform='pandas'):
        self.output = transform


class NumericBinning(BaseEstimator, TransformerMixin):
    """
    Transformer for binning numeric variables into specified ranges.

    Args:
        column (str): Name of the column to bin.
        bins (list of float): List of bin edges for numerical binning.
        labels (list of str): Labels for the bins.
        overwrite (bool, optional): Whether to overwrite the original column. Defaults to True.

    Methods:
        fit(X, y=None):
            Fits the transformer. Returns self (no changes made during fit).

        transform(X):
            Bins the specified numeric column into the defined ranges.

        set_output(transform='pandas'):
            Sets the desired output format for the transformer.

    Example:
        >>> binning = NumericBinning(column='age', bins=[0, 18, 65, 100], labels=['Child', 'Adult', 'Senior'])
        >>> X_transformed = binning.transform(X)
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
            col_name = self.column + '_binned'

        X[col_name] = pd.cut(X[self.column], bins=self.bins, labels=self.labels, right=False)

        return X

    def set_output(self, transform='pandas'):
        self.output = transform
