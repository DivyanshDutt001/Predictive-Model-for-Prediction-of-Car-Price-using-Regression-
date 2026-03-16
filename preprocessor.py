

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder


# Columns that are categorical and need encoding
CATEGORICAL_COLS = ["Brand", "Fuel Type", "Transmission", "Condition", "Model"]

# Column to drop (identifier, not a feature)
DROP_COLS = ["Car ID"]

# Target variable
TARGET_COL = "Price"


def drop_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the Car ID column — it carries no predictive value."""
    df = df.drop(columns=DROP_COLS)
    print(f"Dropped columns : {DROP_COLS}")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label-encode every categorical column.
    Each unique string category is mapped to an integer.
    """
    encoder = LabelEncoder()

    for col in CATEGORICAL_COLS:
        df[col] = encoder.fit_transform(df[col])
        print(f"Encoded         : '{col}'  →  integer labels")

    return df


def split_features_target(df: pd.DataFrame) -> tuple:
    """Return feature matrix X and target series y."""
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    print(f"\nFeatures used   : {list(X.columns)}")
    print(f"Target column   : '{TARGET_COL}'")
    return X, y


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size:    float = 0.20,
    random_state: int   = 42,
) -> tuple:
    """Split into 80 % train and 20 % test subsets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
    )
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing  samples: {len(X_test)}")
    return X_train, X_test, y_train, y_test
