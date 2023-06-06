from sklearn.model_selection import train_test_split
import pandas as pd


def prepare_for_modeling(data, target_column, columns_names = None, drop_columns = None, scaler=None, one_hot_encode=False, convert_binary_to_numeric = False, sampler=None, save_path = None):
    """
    Preprocesses the input data for modeling by performing the following steps:
    1. Renames the columns of the input data to standard names.
    2. Drops unimportant columns.
    3. Handles missing values (TODO: Implement this step).
    4. Splits the dataset into features (X) and target (y).
    5. Splits the dataset into train and test sets.
    6. Scales the feature values to a uniform range.
    TODO: Add all the steps in this function

    Args:
        data (pandas.DataFrame): The input data containing features and target
        target_column (string): Target/Output column
        columns_names (list): New columns names
        drop_columns (list): Unimportant features/columns to be droped
        scaler ():
        one_hot_encode (bool):
        save_path (string): The path where to save the processed/cleaned dataset

    Returns:
        tuple: A tuple containing four elements:
            - X_train_rescaled (numpy.ndarray): Rescaled training features.
            - X_test_rescaled (numpy.ndarray): Rescaled testing features.
            - y_train (pandas.Series): Training target.
            - y_test (pandas.Series): Testing target.
    """

    data = data.copy()

    # Change columns names
    if columns_names is not None:
        data.columns = columns_names

    # Drop unimportant columns
    if drop_columns is not None:
        data.drop(columns=drop_columns, inplace=True)

    # TODO: Check nan values and clean/fix them

    # Save processed/cleaned dataset
    if save_path is not None:
        data.to_csv(save_path + '/data_cleaned.csv')

    # Convert binary columns (yes/no, true/false) to numeric (1/0)
    if convert_binary_to_numeric:
        data = convert_binary_columns_to_numeric(df=data)

    # Splitting the dataset into "features" and "target"
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Sampling Data
    if sampler is not None:
        X, y = sampler.fit_resample(X, y)

    # One hot encode
    if one_hot_encode:
        X = pd.get_dummies(X, columns = get_categorical_columns(X))

    # Splitting the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale the feature values to a uniform range
    if scaler is not None:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)


    return X_train, X_test, y_train, y_test


def convert_binary_columns_to_numeric(df):
    df = df.copy()
    
    for col in df.columns:
        if df[col].dtype == 'object' and len(df[col].value_counts()) == 2:
            values = df[col].unique()
            for value in values:
                new_value = 1 if value == 'yes' else 0
                df[col].replace(value, new_value, inplace=True)
    
    return df


def get_categorical_columns(df):
    categorical_columns = []
    for column in df.columns:
        if df[column].dtype == 'object' and not df[column].str.isnumeric().any():
            categorical_columns.append(column)

    return categorical_columns