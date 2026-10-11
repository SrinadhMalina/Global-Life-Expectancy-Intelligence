import os
import logging
import pandas as pd
from data_preprocessing import preprocess_data
from model_training import prepare_features, train_and_compare_models, optimize_best_model, save_model_assets
from model_evaluation import evaluate_model, explain_model, create_prediction_report

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_pipeline():
    """
    The main orchestrator for the LifeML pipeline.
    """
    # 1. Define Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(BASE_DIR)

    RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'Life Expectancy Data.csv')
    PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'cleaned_life_expectancy.csv')
    MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'best_model.pkl')
    SCALER_PATH = os.path.join(PROJECT_ROOT, 'models', 'scaler.pkl')
    REPORT_PATH = os.path.join(PROJECT_ROOT, 'reports', 'predictions_report.csv')

    logger.info("🚀 Starting LifeML Professional Pipeline...")

    # Stage 1: Preprocessing
    try:
        df_cleaned = preprocess_data(RAW_DATA_PATH, PROCESSED_DATA_PATH)
        logger.info("✅ Preprocessing Complete.")
    except Exception as e:
        logger.error(f"Pipeline failed at preprocessing: {e}")
        return

    # Stage 2: Feature Preparation
    X, y = prepare_features(df_cleaned)

    # Stage 3: Model Comparison & Selection
    results, scaler = train_and_compare_models(X, y)
    best_model_name = max(results, key=results.get)
    logger.info(f"🏆 Winning Model: {best_model_name} with CV R²: {results[best_model_name]:.4f}")

    # Stage 4: Optimization & Final Training
    best_model, X_test, y_test = optimize_best_model(X, y, best_model_name)

    # Save model and scaler for future use
    save_model_assets(best_model, scaler, MODEL_PATH, SCALER_PATH)
    logger.info("✅ Model Optimized and Saved.")

    # Stage 5: Final Evaluation
    r2, mae, rmse, predictions = evaluate_model(best_model, X_test, y_test)
    logger.info(f"📊 Final Test Metrics -> R²: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    # Stage 6: Export for Power BI
    create_prediction_report(y_test, predictions, X_test, REPORT_PATH)
    logger.info(f"✅ Power BI Report generated at {REPORT_PATH}")

    logger.info("🏁 Pipeline Execution Finished Successfully!")

if __name__ == "__main__":
    run_pipeline()
