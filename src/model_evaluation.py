import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import shap

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model(model, X_test, y_test):
    """
    Calculates and logs key regression metrics.
    """
    logger.info("Evaluating model performance...")
    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    logger.info(f"R² Score: {r2:.4f}")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")

    return r2, mae, rmse, predictions

def explain_model(model, X_train):
    """
    Uses SHAP to explain feature importance.
    Professional touch: Moving beyond simple .feature_importances_
    """
    logger.info("Generating SHAP values for model explainability...")

    # SHAP works differently for tree-models vs linear
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)
    except:
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_train)

    return explainer, shap_values

def create_prediction_report(y_test, predictions, X_test, output_path):
    """
    Creates a dataset of Actual vs Predicted values for Power BI.
    Re-constructs the original 'Country' column from one-hot encoded columns.
    """
    logger.info("Creating prediction report for Power BI...")

    report = X_test.copy()

    # Professional touch: Recover the original 'Country' column from one-hot encoded columns
    country_cols = [col for col in X_test.columns if col.startswith('Country_')]
    if country_cols:
        # For each row, find which 'Country_' column is True (1)
        def get_country(row):
            for col in country_cols:
                if row[col] == 1 or row[col] == True:
                    return col.replace('Country_', '')
            return "Unknown"

        report['Country'] = report.apply(get_country, axis=1)
        # Remove the messy one-hot columns so the dashboard is clean
        report = report.drop(columns=country_cols)

    report['Actual_Life_Expectancy'] = y_test.values
    report['Predicted_Life_Expectancy'] = predictions
    report['Error'] = report['Actual_Life_Expectancy'] - report['Predicted_Life_Expectancy']

    report.to_csv(output_path, index=False)
    logger.info(f"Power BI report saved to {output_path}")
    return report
