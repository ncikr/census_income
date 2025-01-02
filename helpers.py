import re

import numpy as np
import pandas as pd
from janitor import clean_names

from custom_transformers import *


def load_data(data_filepath, metadata_filepath, num_features, cat_features):
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

	data[num_features] = data[num_features].astype('int')
	data[cat_features] = data[cat_features].astype('str')

	X = data.drop('income_threshold', axis=1)
	Y = data['income_threshold']

	return X, Y


def load_features():
	num_features = [
		'age',
		'capital_gains',
		'capital_losses',
		'dividends_from_stocks',
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
		'wage_per_hour',
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
		'weeks_worked_in_year',
		'fill_inc_questionnaire_for_veterans_admin',
		'veterans_benefits',
	]

	return num_features, cat_features


def load_transformers():
	transformers = dict()

	# numerical

	transformers['age'] = NumericBinning(
		column='age',
		bins=[0, 16, 67, float('inf')],
		labels=['child', 'working_age', 'retirement_age'],
		overwrite=False,
	)

	# transformers['wage_per_hour'] = NumericBinning(
	# 	column='wage_per_hour',
	# 	bins=[0, 1, 500, float('inf')],
	# 	labels=['0', '0-500', '500+'],
	# 	overwrite=True,
	# )

	# transformers['weeks_worked_in_year'] = NumericBinning(
	# 	column='weeks_worked_in_year',
	# 	bins=[0, 1, 51, float('inf')],
	# 	labels=['0', '0-52', '52'],
	# 	overwrite=True,
	# )

	# categorical

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
