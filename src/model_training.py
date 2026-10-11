import pandas as pd
import numpy as np
import logging
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def prepare_features(df):
    """
    Separates features and target, and handles encoding.
    """
    logger.info("Preparing features and target...")

    # Identify target column (Life expectancy)
    target_col = [col for col in df.columns if 'Life expectancy' in col][0]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Handle categorical variables using One-Hot Encoding
    X = pd.get_dummies(X, drop_first=True)

    return X, y

def train_and_compare_models(X, y):
    """
    Trains multiple regression models and compares their cross-validated R² scores.
    """
    logger.info("Starting model comparison...")

    # Scale features for models like Ridge/Lasso
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "RandomForest": RandomForestRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42)
    }

    results = {}
    for name, model in models.items():
        # Use scaled data for linear models, raw for tree-based
        data = X_scaled if name in ["LinearRegression", "Ridge", "Lasso"] else X
        scores = cross_val_score(model, data, y, cv=5, scoring='r2')
        results[name] = scores.mean()
        logger.info(f"{name} CV R²: {scores.mean():.4f}")

    return results, scaler

def optimize_best_model(X, y, best_model_name):
    """
    Performs hyperparameter tuning on the best performing model.
    """
    logger.info(f"Optimizing {best_model_name} using GridSearchCV...")

    # Standard split for final tuning and validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if best_model_name == "RandomForest":
        model = RandomForestRegressor(random_state=42)
        params = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
    elif best_model_name == "XGBoost":
        model = XGBRegressor(random_state=42)
        params = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    else:
        model = LinearRegression()
        params = {}

    grid = GridSearchCV(model, params, cv=5, scoring='r2', n_jobs=-1)
    grid.fit(X_train, y_train)

    logger.info(f"Best Params for {best_model_name}: {grid.best_params_}")
    return grid.best_estimator_, X_test, y_test

def save_model_assets(model, scaler, path_model, path_scaler):
    """
    Saves the trained model and scaler to disk.
    """
    joblib.dump(model, path_model)
    joblib.dump(scaler, path_scaler)
    logger.info(f"Model and scaler saved to {path_model} and {path_scaler}")
