import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import logging

# Setup logging for professional tracking
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def load_dataset(file_path):
    """
    Loads the dataset from the specified path.
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Dataset loaded successfully from {file_path}. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

def handle_missing_values(df):
    """
    Handles missing values using professional imputation techniques.
    For numerical columns, we use KNNImputer instead of simple mean/median.
    """
    logger.info("Handling missing values...")

    # Identify numerical and categorical columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns

    # 1. Handle Numerical Missing Values using KNN Imputer
    # KNN Imputer looks at similar rows to guess the missing value
    imputer = KNNImputer(n_neighbors=5)
    df[num_cols] = imputer.fit_transform(df[num_cols])

    # 2. Handle Categorical Missing Values
    # For categorical, we use the most frequent value (mode)
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    logger.info("Missing values handled using KNNImputation for numerical and Mode for categorical data.")
    return df

def treat_outliers(df, columns):
    """
    Treats outliers using the Interquartile Range (IQR) method.
    Capping outliers instead of removing them to preserve data volume.
    """
    logger.info("Treating outliers using IQR method...")

    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Capping: Values outside bounds are replaced by the bound value
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

    logger.info(f"Outliers treated for columns: {columns}")
    return df

def preprocess_data(raw_data_path, processed_data_path):
    """
    Main pipeline function to clean the data.
    """
    # Load
    df = load_dataset(raw_data_path)

    # Clean
    df = handle_missing_values(df)

    # Identify numerical columns for outlier treatment (excluding target 'Life expectancy')
    # Assuming the target column name contains 'Life expectancy'
    target_col = [col for col in df.columns if 'Life expectancy' in col][0]
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols.remove(target_col)

    df = treat_outliers(df, num_cols)

    # Save processed data
    df.to_csv(processed_data_path, index=False)
    logger.info(f"Processed data saved to {processed_data_path}")

    return df
