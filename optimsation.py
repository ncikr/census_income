import mlflow
import numpy as np
import optuna
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from custom_transformers import DropColumns
from helpers import load_clean_data, load_features

num_features, cat_features = load_features()

X_train, y_train = load_clean_data(
    data_filepath='data/census_income_learn.csv',
    metadata_filepath='./data/census_income_metadata.txt',
    num_features=num_features,
    cat_features=cat_features,
)

X_test, y_test = load_clean_data(
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


# imputing transformer
imputer = ColumnTransformer(
    [
        ('num_features', SimpleImputer(strategy='median'), make_column_selector(dtype_include=np.number)),
        ('cat_features', SimpleImputer(strategy='most_frequent'), make_column_selector(dtype_include=object)),
    ],
    remainder='passthrough',
    verbose_feature_names_out=False,
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


def objective(trial):
    # Define the hyperparameters to tune
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)

    # build pipeline
    pipeline = ImbPipeline(
        steps=[
            ('drop_cols', DropColumns(cols_to_drop)),  # separate transformer for readability
            ('impute', imputer),  # impute before binning
            ('scaling_encoding', scaling_encoding),  # scaling and OHE
            ('smote', SMOTE(random_state=42)),
            (
                'classifier',
                GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42,
                ),
            ),
        ],
    ).set_output(transform='pandas')

    # Start MLflow run
    with mlflow.start_run(nested=True):
        # Evaluate the pipeline using cross-validation
        scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring='f1', n_jobs=-1)
        mean_f1 = np.mean(scores)

        # Log parameters and metrics to MLflow
        mlflow.log_param('n_estimators', n_estimators)
        mlflow.log_param('max_depth', max_depth)
        mlflow.log_param('min_samples_split', min_samples_split)
        mlflow.log_metric('mean_f1', mean_f1)

        # Optionally, log the pipeline (model structure)
        mlflow.sklearn.log_model(pipeline, 'model')

    # Return the mean F1 score for Optuna to optimize
    return mean_f1


# create optuna study and optimise
study = optuna.create_study(direction='maximize')  # Maximize F1 score

# Start MLflow parent run
with mlflow.start_run(run_name='SMOTE_Optuna_GradientBoostClassifer'):
    # Optimize the study
    study.optimize(objective, n_trials=20)

    # Log the best trial to MLflow
    mlflow.log_param('best_n_estimators', study.best_trial.params['n_estimators'])
    mlflow.log_param('best_max_depth', study.best_trial.params['max_depth'])
    mlflow.log_param('best_min_samples_split', study.best_trial.params['min_samples_split'])
    mlflow.log_metric('best_mean_f1', study.best_value)
