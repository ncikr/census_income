# from sklearn.compose import ColumnTransformer
import numpy as np
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from custom_transformers import DropColumns
from helpers import load_data, load_features, load_transformers

num_features, cat_features = load_features()

X_train, y_train = load_data(
	data_filepath='data/census_income_learn.csv',
	metadata_filepath='./data/census_income_metadata.txt',
	num_features=num_features,
	cat_features=cat_features,
)

X_test, y_test = load_data(
	data_filepath='data/census_income_test.csv',
	metadata_filepath='./data/census_income_metadata.txt',
	num_features=num_features,
	cat_features=cat_features,
)

# set features to remove from data (based on EDA)
cols_to_drop = [
	'family_members_under_18',
	'fill_inc_questionnaire_for_veterans_admin',
	'instance_weight',
	'year',
]

# load binning transformers
custom_transformers = load_transformers()

binning = ColumnTransformer(
	[(f'{feature}_binning', custom_transformers[feature], [feature]) for feature in custom_transformers],
	remainder='passthrough',
)

# create separate imputing transformer to apply before binning
imputer = ColumnTransformer(
	[
		('num_features', SimpleImputer(strategy='median'), make_column_selector(dtype_include=np.number)),
		('cat_features', SimpleImputer(strategy='most_frequent'), make_column_selector(dtype_include=object)),
	],
	remainder='passthrough',
	verbose_feature_names_out=False,  # to enable binning afterwards
)

# scaling and OHE
preprocessing = ColumnTransformer(
	[
		('num_features', StandardScaler(), make_column_selector(dtype_include=np.number)),
		(
			'cat_features',
			OneHotEncoder(handle_unknown='ignore', sparse_output=False),
			make_column_selector(dtype_include=object),
		),
	]
)

# build pipeline
pipeline = ImbPipeline(
	steps=[
		('drop_cols', DropColumns(cols_to_drop)),  # separate transformer for readability
		('impute', imputer),  # impute before binning
		('binning', binning),  # bin select variables to reduce likelihood of overfitting
		('preprocessing', preprocessing),  # scaling and OHE
		# ('smote', SMOTE(random_state=42)),
	],
).set_output(transform='pandas')  # required to enable column-based pre-processing after binning

pipeline.fit_transform(X_train)
