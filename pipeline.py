# from sklearn.compose import ColumnTransformer
import mlflow
import numpy as np
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from custom_transformers import DropColumns
from helpers import load_base_models, load_data, load_features, load_transformers

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

# create dummy transformer to test effect of binning
no_binning = ColumnTransformer(
	transformers=[],
	remainder='passthrough',
	verbose_feature_names_out=False,  # to enable binning afterwards
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
scaling_encoding = ColumnTransformer(
	[
		('num_features', StandardScaler(), make_column_selector(dtype_include=np.number)),
		(
			'cat_features',
			OneHotEncoder(handle_unknown='ignore', sparse_output=False),
			make_column_selector(dtype_include=object),
		),
	]
)

mlflow.set_experiment('census_income_classification')

# load models to test
models = load_base_models()

for binning_enabled in [True, False]:
	# build pipeline
	preprocessing = ImbPipeline(
		steps=[
			('drop_cols', DropColumns(cols_to_drop)),  # separate transformer for readability
			('impute', imputer),  # impute before binning
			(
				'binning',
				binning if binning_enabled else no_binning,
			),  # bin select variables to reduce likelihood of overfitting
			('scaling_encoding', scaling_encoding),  # scaling and OHE
			# ('smote', SMOTE(random_state=42)), # disable oversamping for now
		],
	).set_output(transform='pandas')  # required to enable column-based pre-processing after binning

	# run each model and log metrics
	for model_name, model in models.items():
		with mlflow.start_run(run_name=f'{model_name}_binning_{binning_enabled}'):
			mlflow.log_param('model_name', model_name)
			mlflow.log_param('binning_enabled', binning_enabled)

			# append classifer to pipeline
			pipeline = make_pipeline(*preprocessing, model)

			# fit pipeline
			pipeline.fit(X_train, y_train)

			# evaluate
			y_pred = pipeline.predict(X_test)
			y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
			f1 = f1_score(y_test, y_pred)
			roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

			mlflow.log_metric('test_f1_score', f1)
			if roc_auc is not None:
				mlflow.log_metric('test_roc_auc', roc_auc)

			# cross-validation

			# cv_f1_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1')
			# cv_roc_auc_scores = (
			# 	cross_val_score(pipeline, X_train, y_train, cv=5, scoring=make_scorer(roc_auc_score, needs_proba=True))
			# 	if hasattr(model, 'predict_proba')
			# 	else None
			# )
			# mean_cv_f1 = np.mean(cv_f1_scores)
			# mean_cv_roc_auc = np.mean(cv_roc_auc_scores) if cv_roc_auc_scores is not None else None

			# mlflow.log_metric('mean_cv_f1_score', mean_cv_f1)
			# if mean_cv_roc_auc is not None:
			# 	mlflow.log_metric('mean_cv_roc_auc_score', mean_cv_roc_auc)

			# log the pipeline
			mlflow.sklearn.log_model(pipeline, artifact_path='pipeline')

			print(
				f"Logged {model_name} (Binning: {binning_enabled}) to MLflow: "
				f"Test F1 = {f1:.4f}, Test ROC AUC = {roc_auc:.4f if roc_auc else 'N/A'}, "
			)
