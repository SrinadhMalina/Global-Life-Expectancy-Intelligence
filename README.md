# 🌍 LifeML: Global Life Expectancy Predictive Analytics

An end-to-end professional machine learning pipeline and business intelligence dashboard designed to predict and analyze Life Expectancy based on WHO health and socio-economic indicators.

## 🎯 Project Objective
The goal of this project was to move beyond simple predictive modeling to create a **production-ready ML pipeline**. By analyzing data from 190+ countries, the project identifies the key drivers of longevity and provides a high-accuracy tool for health forecasting.

## 🚀 Professional Highlights
Unlike a standard data science notebook, this project implements industry-standard software engineering practices:
- **Modular Architecture**: Separated logic into `preprocessing`, `training`, and `evaluation` modules for maintainability.
- **Advanced Data Cleaning**: Implemented **KNN Imputation** for missing values and **IQR Capping** to handle outliers without losing data volume.
- **Model Tournament**: Conducted a competitive analysis between **Linear Regression, Ridge, Lasso, Random Forest, and XGBoost** using 5-fold Cross-Validation.
- **Hyperparameter Optimization**: Utilized **GridSearchCV** to fine-tune the winning model.
- **Model Explainability**: Integrated **SHAP (SHapley Additive exPlanations)** to move from a "black box" model to interpretable insights.

## 📊 Results & Performance
The final model achieved exceptional predictive power:
- **Winning Model**: Random Forest Regressor
- **Test $R^2$ Score**: **0.9657** (Explains ~96.6% of the variance in life expectancy)
- **Mean Absolute Error (MAE)**: **1.10 years** (Predictions are on average only 1.1 years off)
- **RMSE**: **1.72 years**

## 📈 Executive Dashboard (Power BI)
The model's output is fed into a Power BI dashboard designed for non-technical stakeholders:
- **Executive Summary**: High-level KPIs and global accuracy metrics.
- **Geospatial Analysis**: Interactive map visualizing predicted longevity by country.
- **Feature Importance**: Visual breakdown of how factors like Schooling, GDP, and Adult Mortality drive life expectancy.
- **Country Deep-Dive**: Dynamic filtering for individual country health profiles.

## 🛠️ Tech Stack
- **Core**: Python 3.x, Scikit-Learn, Pandas, NumPy
- **Optimization**: XGBoost, GridSearchCV
- **Explainability**: SHAP
- **Visualization**: Power BI, Matplotlib, Seaborn

## 📂 Project Structure
```text
LifeML/
├── data/               # Raw and processed datasets (Git-ignored)
├── src/                # Production Python modules
│   ├── data_preprocessing.py # Cleaning & Imputation
│   ├── model_training.py     # Tournament & Optimization
│   ├── model_evaluation.py   # Metrics & SHAP
│   └── pipeline.py           # Main Orchestrator
├── notebooks/          # EDA & Prototyping
├── models/             # Serialized models (.pkl)
└── reports/            # Power BI Source & Screenshots
```

## ⚙️ How to Run
1. **Clone the repo** and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. **Prepare Data**: Place the WHO dataset in `data/raw/Life Expectancy Data.csv`.
3. **Execute Pipeline**:
   ```bash
   python src/pipeline.py
   ```
4. **Visualize**: Load `reports/predictions_report.csv` into Power BI.

---
**Author:** [Your Name] | **Role:** Data Scientist / ML Engineer
