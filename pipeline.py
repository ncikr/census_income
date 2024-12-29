# from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from helpers import load_data, load_features, load_transformers

X_train, y_train = load_data(
	data_filepath='data/census_income_learn.csv',
	metadata_filepath='./data/census_income_metadata.txt',
)

X_test, y_test = load_data(
	data_filepath='data/census_income_test.csv',
	metadata_filepath='./data/census_income_metadata.txt',
)

num_features, cat_features = load_features()

custom_transformers = load_transformers()

# build pipeline to bin features based on eda
binning_pipeline = Pipeline(
	steps=[
		('age_binning', custom_transformers['age']),
		('wage_per_hour_binning', custom_transformers['wage_per_hour']),
		('weeks_worked_in_year', custom_transformers['weeks_worked_in_year']),
		('education', custom_transformers['education']),
		('marital_stat', custom_transformers['marital_stat']),
		('race', custom_transformers['race']),
		('reason_for_unemployment', custom_transformers['reason_for_unemployment']),
		('full_or_part_time_employment_stat', custom_transformers['full_or_part_time_employment_stat']),
		('citizenship', custom_transformers['citizenship']),
	]
)

# standard preprocessing
num_pipeline = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())

cat_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore'))

preprocessing = ColumnTransformer(
	[('num_features', num_pipeline, num_features), ('cat_features', cat_pipeline, cat_features)]
)

# build pipeline with SMOTE
pipeline = ImbPipeline(
	steps=[
		('binning', binning_pipeline),
		('preprocessing', preprocessing),
		('smote', SMOTE(random_state=42)),
	]
)

pipeline
