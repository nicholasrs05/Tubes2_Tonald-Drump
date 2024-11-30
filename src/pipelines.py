import logging
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import shuffle
from sklearn.preprocessing import PolynomialFeatures, RobustScaler

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def handleMissingValue(df, numeric_columns=None, categorical_columns=None):
    numeric_imputer = SimpleImputer(strategy="mean")
    categorical_imputer = SimpleImputer(strategy="most_frequent")

    df[numeric_columns] = numeric_imputer.fit_transform(df[numeric_columns])
    df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])

    return df


def handleOutlier(df, numeric_columns=None, categorical_columns=None):
    """
    Enhanced outlier handling with winsorization
    """
    logger.info("Handling outliers...")
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df[column] = np.clip(df[column], lower_bound, upper_bound)

    logger.info("Outliers handled successfully.")
    return df


def removeDuplicates(df):
    """
    Remove duplicate rows and reset index
    """
    logger.info("Removing duplicate rows...")
    original_size = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info(f"Removed {original_size - len(df)} duplicate rows.")
    return df


def engineerFeature(df, numeric_columns=None, categorical_columns=None):
    """
    Enhanced feature engineering with additional checks
    """
    logger.info("Engineering features...")
    try:
        # Check for duplicates in the original DataFrame
        if df.columns.duplicated().any():
            raise ValueError(
                f"Duplicate column names found: {df.columns[df.columns.duplicated()]}"
            )

        # Polynomial Features
        poly_features = ["sbytes", "dbytes", "sload", "dload"]
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        poly_features_transformed = poly.fit_transform(df[poly_features])
        poly_df = pd.DataFrame(
            poly_features_transformed,
            columns=[
                f"poly_{col}" for col in poly.get_feature_names_out(poly_features)
            ],
        )

        # Concatenate while avoiding duplicate column names
        df = pd.concat([df, poly_df], axis=1)
        if df.columns.duplicated().any():
            raise ValueError(
                f"Duplicate column names found after concatenation: {df.columns[df.columns.duplicated()]}"
            )

        # Add custom engineered features
        df["network_pkt_loss_ratio"] = df["sloss"] / (df["spkts"] + 1e-6)
        df["network_efficiency"] = (df["sbytes"] + df["dbytes"]) / (
            df["spkts"] + df["dpkts"] + 1e-6
        )
        df["network_byte_ratio"] = df["sbytes"] / (df["dbytes"] + 1e-6)
        df["network_pkt_ratio"] = df["spkts"] / (df["dpkts"] + 1e-6)
        df["network_load_ratio"] = df["sload"] / (df["dload"] + 1e-6)
        df["network_loss_diff"] = df["sloss"] - df["dloss"]

        df["network_pkt_rate"] = (df["spkts"] + df["dpkts"]) / (df["dur"] + 1e-6)
        df["network_byte_rate"] = (df["sbytes"] + df["dbytes"]) / (df["dur"] + 1e-6)

        df["network_response_ratio"] = df["response_body_len"] / (df["dbytes"] + 1e-6)

        # Log-transform selected columns
        log_cols = ["sbytes", "dbytes", "sload", "dload"]
        df[log_cols] = df[log_cols].apply(np.log1p)
    except Exception as e:
        logger.error(f"Error during feature engineering: {e}")
        raise

    logger.info("Features engineered successfully.")
    return df


def featureScaling(df, numeric_columns=None, categorical_columns=None):
    """
    Enhanced feature scaling with robust scaler option
    """
    logger.info("Scaling features...")
    try:
        columns = df.select_dtypes(include=["int64", "float64"]).columns
        scaler = RobustScaler()
        df[columns] = scaler.fit_transform(df[columns])
    except Exception as e:
        logger.error(f"Error during feature scaling: {e}")
        raise

    logger.info("Features scaled successfully.")
    return df


def encodeFeatures(df, columns=None, target_column="attack_cat"):
    """
    Enhanced feature encoding with one-hot encoding option
    """
    logger.info("Encoding features...")
    from sklearn.preprocessing import OneHotEncoder

    columns = columns[columns != target_column]

    try:
        if columns is None:
            columns = df.select_dtypes(include=["object"]).columns

        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        categorical_encoded = ohe.fit_transform(df[columns])
        categorical_columns = ohe.get_feature_names_out(columns)

        encoded_df = pd.DataFrame(
            categorical_encoded, columns=categorical_columns, index=df.index
        )

        df = df.drop(columns=columns)
        df = pd.concat([df, encoded_df], axis=1)
    except Exception as e:
        logger.error(f"Error during feature encoding: {e}")
        raise

    logger.info("Features encoded successfully.")
    return df


def handleClassImbalance(df, target_column="attack_cat", random_state=42):
    """
    Enhanced class imbalance handling with improved performance
    """
    logger.info("Handling class imbalance...")
    try:
        y = df[target_column]
        X = df.drop(target_column, axis=1)

        smote = SMOTE(sampling_strategy="auto", random_state=random_state)
        rus = RandomUnderSampler(sampling_strategy="auto", random_state=random_state)

        X_resampled, y_resampled = smote.fit_resample(X, y)
        X_resampled, y_resampled = rus.fit_resample(X_resampled, y_resampled)

        # Create the balanced DataFrame more efficiently
        balanced_df = pd.concat(
            [
                pd.DataFrame(X_resampled, columns=X.columns),
                pd.Series(y_resampled, name=target_column),
            ],
            axis=1,
        )
    except Exception as e:
        logger.error(f"Error during class imbalance handling: {e}")
        raise

    logger.info("Class imbalance handled successfully.")
    
    balanced_df = shuffle(balanced_df, random_state=random_state)
    
    return balanced_df


def create_preprocessing_pipeline(
    df, numeric_columns=None, categorical_columns=None, target_column="attack_cat"
):
    """
    Create a preprocessing pipeline using the existing functions
    """
    logger.info("Starting preprocessing pipeline...")
    try:
        df = handleMissingValue(df, numeric_columns, categorical_columns)
        df = removeDuplicates(df)
        df = handleOutlier(df, numeric_columns, categorical_columns)
        df = engineerFeature(df, numeric_columns, categorical_columns)
        df = encodeFeatures(df, categorical_columns)
        df = featureScaling(df, numeric_columns, categorical_columns)
        df = handleClassImbalance(df, target_column)
    except Exception as e:
        logger.error(f"Error during preprocessing pipeline: {e}")
        raise

    logger.info("Preprocessing pipeline completed successfully.")
    return df
