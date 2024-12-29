# from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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

binning_pipeline.fit_transform(X_train)
