import re
import string

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from janitor import clean_names
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, f1_score, make_scorer, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score

from custom_transformers import CategoricalBinning, NumericBinning


def load_clean_data(data_filepath, metadata_filepath, num_features, cat_features):
    """
    Loads and cleans the dataset, preparing it for modeling.

    Args:
        data_filepath (str): Path to the dataset file.
        metadata_filepath (str): Path to the metadata file containing column descriptions.
        num_features (list of str): List of numerical feature names.
        cat_features (list of str): List of categorical feature names.

    Returns:
        tuple: A tuple (X, Y) where:
            - X (pd.DataFrame): Features after cleaning and preprocessing.
            - Y (pd.Series): Target variable (`income_threshold`).

    Raises:
        FileNotFoundError: If the specified data or metadata file does not exist.
        ValueError: If feature names cannot be extracted from metadata.
    """
    # load feature names from metadata
    metadata_file = open(metadata_filepath, 'r')
    metadata_lines = metadata_file.readlines()
    columns = [re.findall(r"(\b[a-z\s'1-9-]+\b):", line)[0] for line in metadata_lines[142:]]
    columns.append('income_threshold')

    # load data and format cols
    data = (
        pd.read_csv(data_filepath, names=columns, index_col=False)
        .pipe(clean_names)
        .assign(income_threshold=lambda x: np.where(x['income_threshold'] == ' 50000+.', 1, 0))
    )

    # cleaning
    data['weeks_worked_in_year'] = data['weeks_worked_in_year'].replace('other', 30)  # to enable numerical binning
    data[num_features] = data[num_features].astype('int')
    data[cat_features] = data[cat_features].astype('str')

    # prevent problems with punctuation in feature names
    data[cat_features] = data[cat_features].replace(string.punctuation, '')

    X = data.drop('income_threshold', axis=1)
    Y = data['income_threshold']

    return X, Y


def load_features():
    """
    Defines numerical and categorical features for the dataset.

    Returns:
        tuple: A tuple (num_features, cat_features) where:
            - num_features (list of str): List of numerical feature names.
            - cat_features (list of str): List of categorical feature names.
    """
    num_features = [
        'age',
        'wage_per_hour',
        'capital_gains',
        'capital_losses',
        'dividends_from_stocks',
        'weeks_worked_in_year',
    ]

    cat_features = [
        'class_of_worker',
        'detailed_industry_recode',
        'detailed_occupation_recode',
        'education',
        'enroll_in_edu_inst_last_wk',
        'marital_stat',
        'major_industry_code',
        'major_occupation_code',
        'race',
        'hispanic_origin',
        'sex',
        'member_of_a_labor_union',
        'reason_for_unemployment',
        'full_or_part_time_employment_stat',
        'tax_filer_stat',
        'region_of_previous_residence',
        'state_of_previous_residence',
        'detailed_household_and_family_stat',
        'detailed_household_summary_in_household',
        'migration_code_change_in_msa',
        'migration_code_change_in_reg',
        'migration_code_move_within_reg',
        'live_in_this_house_1_year_ago',
        'num_persons_worked_for_employer',
        'family_members_under_18',
        'country_of_birth_father',
        'country_of_birth_mother',
        'country_of_birth_self',
        'citizenship',
        'own_business_or_self_employed',
        'fill_inc_questionnaire_for_veterans_admin',
        'veterans_benefits',
    ]

    return num_features, cat_features


def load_transformers():
    """
    Loads feature transformers for binning numerical and categorical variables.

    Returns:
        dict: A dictionary of transformers, where each key is a feature name and the value is a transformer object.
    """
    transformers = dict()

    # numerical transformers

    transformers['age'] = NumericBinning(
        column='age',
        bins=[0, 16, 67, float('inf')],
        labels=['child', 'working_age', 'retirement_age'],
        overwrite=False,
    )

    transformers['wage_per_hour'] = NumericBinning(
        column='wage_per_hour',
        bins=[0, 1, 500, float('inf')],
        labels=['0', '0-500', '500+'],
        overwrite=True,
    )

    transformers['weeks_worked_in_year'] = NumericBinning(
        column='weeks_worked_in_year',
        bins=[0, 1, 51, float('inf')],
        labels=['0', '0-52', '52'],
        overwrite=True,
    )

    # categorical transformers

    transformers['education'] = CategoricalBinning(
        column='education',
        bins={
            'Child': ['Children'],
            'High School Dropout': [
                '7th and 8th grade',
                '10th grade',
                '11th grade',
                '9th grade',
            ],
            'High School Graduate': ['High school graduate'],
            'College Dropout': ['Some college but no degree'],
            'Higher Education': [
                'Bachelors degree(BA AB BS)',
                'Masters degree(MA MS MEng MEd',
                'Associates degree-occup /voca',
            ],
            'Other': ['other'],
        },
        labels=[
            'Child',
            'High School Dropout',
            'High School Graduate',
            'College Dropout',
            'Higher Education',
            'Other',
        ],
        overwrite=True,
    )

    transformers['marital_stat'] = CategoricalBinning(
        column='marital_stat',
        bins={
            'Child': ['Children'],
            'High School Dropout': [
                '7th and 8th grade',
                '10th grade',
                '11th grade',
                '9th grade',
            ],
            'High School Graduate': ['High school graduate'],
            'College Dropout': ['Some college but no degree'],
            'Higher Education': [
                'Bachelors degree(BA AB BS)',
                'Masters degree(MA MS MEng MEd',
                'Associates degree-occup /voca',
            ],
            'Other': ['other'],
        },
        labels=[
            'Child',
            'High School Dropout',
            'High School Graduate',
            'College Dropout',
            'Higher Education',
            'Other',
        ],
        overwrite=True,
    )

    transformers['race'] = CategoricalBinning(
        column='race',
        bins={
            'White': ['White'],
            'Black': ['Black'],
            'Other': [
                'Asian or Pacific Islander',
                'Other',
                'Amer Indian Aleut or Eskimo',
            ],
        },
        labels=['White', 'Black', 'Other'],
        overwrite=True,
    )

    transformers['reason_for_unemployment'] = CategoricalBinning(
        column='reason_for_unemployment',
        bins={
            'NIU': ['Not in universe'],
            'IU': [
                'Other job loser',
                'Re-entrant',
                'Job loser - on layoff',
                'Job leaver',
                'New entrant',
            ],
        },
        labels=['NIU', 'IU'],
        overwrite=True,
    )

    transformers['full_or_part_time_employment_stat'] = CategoricalBinning(
        column='full_or_part_time_employment_stat',
        bins={
            'children or af': ['Children or Armed Forces'],
            'not in labor force': ['Not in labor force'],
            'full-time': ['Full-time schedules'],
            'other': [
                'PT for non-econ reasons usual',
                'Unemployed full-time',
                'PT for econ reasons usually P',
                'Unemployed part- time',
                'PT for econ reasons usually F',
            ],
        },
        labels=['children or af', 'not in labor force', 'full-time', 'other'],
        overwrite=True,
    )

    transformers['citizenship'] = CategoricalBinning(
        column='citizenship',
        bins={
            'native': [
                'Native- Born in the United St',
                'Native- Born abroad of Americ',
                'Native- Born in Puerto Rico o',
            ],
            'non-native': [
                'Foreign born- Not a citizen o',
                'Foreign born- U S citizen by',
            ],
        },
        labels=['native', 'non-native'],
        overwrite=True,
    )

    return transformers


def load_base_models():
    """
    Loads base machine learning models for binary classification.

    Returns:
        dict: A dictionary of models, where each key is the model name and the value is a scikit-learn model object.
    """
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        # 'SVC': SVC(probability=True, random_state=42), # training taking too long for this exercise
        # 'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), # training taking too long for this exercise
    }

    return models


def evaluate_pipeline(
    model_name, model, pipeline, X_train, y_train, X_test, y_test, cross_val=False, mlflow_logging=True
):
    """
    Evaluates a machine learning pipeline, computing metrics and generating plots.

    Args:
        model_name (str): Name of the model (used for saving plots).
        model (sklearn.base.BaseEstimator): Trained model object.
        pipeline (sklearn.pipeline.Pipeline): Pipeline containing preprocessing and the model.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.Series): Testing target.
        cross_val (bool, optional): Whether to perform cross-validation. Default is False.
        mlflow_logging (bool, optional): Whether to log metrics and artifacts to MLflow. Default is True.

    Returns:
        tuple: A tuple (eval_metrics, pr_data, roc_data) where:
            - eval_metrics (dict): Dictionary of evaluation metrics (e.g., F1, ROC-AUC).
            - pr_data (dict): Precision-recall curve data.
            - roc_data (dict): ROC curve data.

    Saves:
        - Precision-recall curve plot.
        - Precision, recall, and F1 score vs. threshold plot.

    Raises:
        ValueError: If the pipeline or model is not properly fitted.
    """
    # precision recall
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

    # find optimal f1
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]

    # pr curve
    pr_auc = auc(recalls, precisions)
    plt.figure(figsize=(5, 5))
    plt.plot(recalls, precisions, label=f'PR Curve (AUC={pr_auc})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    pr_curve_path = f'./plots/pr_curve_{model_name}.png'
    plt.savefig(pr_curve_path)
    plt.close()

    # plot pr-f1 against threshold
    plt.figure(figsize=(5, 5))
    plt.plot(thresholds, precisions[:-1], label='Precision', color='blue')
    plt.plot(thresholds, recalls[:-1], label='Recall', color='green')
    plt.plot(thresholds, f1_scores[:-1], label='F1 Score', color='red')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision, Recall, and F1 Score vs. Threshold')
    plt.legend(loc='best')
    plt.grid(True)
    pr_threshold_path = f'./plots/pr_threshold_plot_{model_name}.png'
    plt.savefig(pr_threshold_path)
    plt.close()

    # cross-validation
    if cross_val:
        cv_f1_scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring='f1')
        cv_roc_auc_scores = (
            cross_val_score(pipeline, X_train, y_train, cv=5, scoring=make_scorer(roc_auc_score))
            if hasattr(model, 'predict_proba')
            else None
        )
        mean_cv_f1 = np.mean(cv_f1_scores)
        mean_cv_roc_auc = np.mean(cv_roc_auc_scores) if cv_roc_auc_scores is not None else None
    else:
        mean_cv_f1 = None
        mean_cv_roc_auc = None

    # save metrics in dictionaries
    eval_metrics = {
        'f1': f1,
        'roc_auc': roc_auc,
        'optimal_threshold': optimal_threshold,
        'optimal_f1': optimal_f1,
        'mean_cv_f1': mean_cv_f1,
        'mean_cv_roc_auc': mean_cv_roc_auc,
    }

    pr_data = {'precisions': precisions, 'recalls': recalls, 'thresholds': thresholds}

    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_prob)

    roc_data = {'fpr': false_positive_rate, 'tpr': true_positive_rate}

    # mlflow logging
    if mlflow_logging:
        mlflow.log_metric('test_f1_score', f1)
        mlflow.log_metric('test_roc_auc', roc_auc)

        if optimal_threshold is not None:
            mlflow.log_param('optimal_threshold', optimal_threshold)
            mlflow.log_param('optimal_f1', optimal_f1)

        if cross_val:
            mlflow.log_metric('mean_cv_f1_score', mean_cv_f1)
            mlflow.log_metric('mean_cv_roc_auc', mean_cv_roc_auc)

        mlflow.log_artifact(pr_curve_path, artifact_path='plots')
        mlflow.log_artifact(pr_threshold_path, artifact_path='plots')

        mlflow.sklearn.log_model(pipeline, artifact_path='pipeline')

    return eval_metrics, pr_data, roc_data
