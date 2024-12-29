import re

import numpy as np
import pandas as pd
from janitor import clean_names

from custom_transformers import *


def load_data(data_filepath, metadata_filepath):
    # load feature names from metadata
    metadata_file = open(metadata_filepath, "r")
    metadata_lines = metadata_file.readlines()
    columns = [
        re.findall(r"(\b[a-z\s'1-9-]+\b):", line)[0] for line in metadata_lines[142:]
    ]
    columns.append("income_threshold")

    # load data and format cols
    data = (
        pd.read_csv(data_filepath, names=columns, index_col=False)
        .pipe(clean_names)
        .assign(
            income_threshold=lambda x: np.where(
                x["income_threshold"] == " 50000+.", 1, 0
            )
        )
    )

    X = data.drop("income_threshold", axis=1)
    Y = data["income_threshold"]

    return X, Y


def load_features():
    num_features = [
        "age",
        "capital_gains",
        "capital_losses",
        "dividends_from_stocks",
        "weeks_worked_in_year",
    ]

    cat_features = [
        "class_of_worker",
        "detailed_industry_recode",
        "detailed_occupation_recode",
        "education",
        "enroll_in_edu_inst_last_wk",
        "marital_stat",
        "major_industry_code",
        "major_occupation_code",
        "wage_per_hour",
        "race",
        "hispanic_origin",
        "sex",
        "member_of_a_labor_union",
        "reason_for_unemployment",
        "full_or_part_time_employment_stat",
        "tax_filer_stat",
        "region_of_previous_residence",
        "state_of_previous_residence",
        "detailed_household_and_family_stat",
        "detailed_household_summary_in_household",
        "migration_code_change_in_msa",
        "migration_code_change_in_reg",
        "migration_code_move_within_reg",
        "live_in_this_house_1_year_ago",
        "num_persons_worked_for_employer",
        "family_members_under_18",
        "country_of_birth_father",
        "country_of_birth_mother",
        "country_of_birth_self",
        "citizenship",
        "own_business_or_self_employed",
        "weeks_worked_in_year",
        "fill_inc_questionnaire_for_veterans_admin",
        "veterans_benefits",
    ]

    return num_features, cat_features


def load_transformers():
    transformers = dict()

    transformers["age"] = NumericBinning(
        column="age",
        bins=[0, 16, 67, float("inf")],
        labels=["child", "working_age", "retirement_age"],
        overwrite=False,
    )

    transformers["wage_per_hour"] = NumericBinning(
        column="wage_per_hour",
        bins=[0, 1, 500, float("inf")],
        labels=["0", "0-500", "500+"],
        overwrite=True,
    )

    transformers["weeks_worked_in_year"] = NumericBinning(
        column="weeks_worked_in_year",
        bins=[0, 1, 51, float("inf")],
        labels=["0", "0-52", "52"],
        overwrite=True,
    )

    return transformers
