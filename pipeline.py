from helpers import load_data

X_train, y_train = load_data(
    data_filepath="data/census_income_learn.csv",
    metadata_filepath="./data/census_income_metadata.txt",
)

X_test, y_test = load_data(
    data_filepath="data/census_income_test.csv",
    metadata_filepath="./data/census_income_metadata.txt",
)
