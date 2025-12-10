# YouTube Video Engagement Prediction

This project analyzes YouTube video data to predict view counts and engagement metrics using **scraped data** and **API data**. It trains regression models and visualizes feature importance, actual vs predicted views, and engagement trends.

---

## Project Structure

Project1-CMPSC445/
│
├─ Scrape_Model.py # Regression on scraped video data
├─ API_Model.py # Regression on API video data
├─ Youtube_Data.csv # Scraped dataset
├─ youtube_videos.csv # API dataset
├─ rf_feature_importance.png
├─ rf_actual_vs_pred.png
├─ xgb_feature_importance.png
├─ xgb_actual_vs_pred.png
├─ engagement_*.png # Engagement trend plots
└─ README.md

yaml
Copy code

---

## Requirements

- Python 3.10+
- Packages:
  ```bash
  pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap
How to Run
Scraped Data Model

bash
Copy code
python Scrape_Model.py
Trains Linear Regression or chosen models on scraped YouTube data.

Generates PNG visualizations for:

Feature importance

Actual vs Predicted views

Engagement trends

API Data Model

bash
Copy code
python API_Model.py
Trains Random Forest and XGBoost models on API video data.

Generates PNG visualizations:

Feature importance for each model

Actual vs Predicted views

Engagement trends (likes, comments, duration, subscribers)

Outputs
PNG plots saved in the project directory.

Selected features and model performance printed in the terminal.

Predictions and visualizations can be compared between scraped and API datasets.
