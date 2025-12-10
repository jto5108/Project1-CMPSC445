import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)

# ----------------------------
# Load CSV
# ----------------------------
df = pd.read_csv("Youtube_Data.csv")

# Helper function to standardize column names
df.columns = df.columns.str.lower()

# ----------------------------
# Preprocessing
# ----------------------------
def convert_views(view_str):
    if pd.isnull(view_str):
        return 0
    view_str = str(view_str).lower().replace("views","").strip()
    if "k" in view_str: return int(float(view_str.replace("k",""))*1_000)
    if "m" in view_str: return int(float(view_str.replace("m",""))*1_000_000)
    try: return int(float(view_str))
    except: return 0

def convert_comments(comment_str):
    if pd.isnull(comment_str):
        return 0
    comment_str = str(comment_str).replace("Comments","").replace(",","").strip()
    try: return int(float(comment_str))
    except: return 0

def convert_date(date_str):
    if pd.isnull(date_str): return 0
    parts = str(date_str).split()
    try:
        val = int(parts[0])
        unit = parts[1]
        if unit in ["year","years"]: return val*365
        if unit in ["month","months"]: return val*30
        if unit in ["week","weeks"]: return val*7
        if unit in ["day","days"]: return val
        return 0
    except: return 0

# Apply conversions
df['views'] = df['views'].apply(convert_views)
df['comments'] = df['comments'].apply(convert_comments)
df['uploaddate'] = df['uploaddate'].apply(convert_date)

# TF-IDF for Title and FirstTag
def tfidf_features(df, column):
    df[column] = df[column].fillna("no_tag")
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform(df[column])
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"{column}_{c}" for c in vectorizer.get_feature_names_out()])
        df = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
    except:
        pass
    return df

df = tfidf_features(df, "title")
df = tfidf_features(df, "firsttag")

# Numerical features
df['title_length'] = df['title'].apply(lambda x: len(str(x)))
df['title_word_count'] = df['title'].apply(lambda x: len(str(x).split()))
df['first_tag_length'] = df['firsttag'].apply(lambda x: len(str(x)))

# One-hot encode category
df = pd.get_dummies(df, columns=["category"], drop_first=True)

# Remove missing target
df = df.dropna(subset=['views'])

# Log transform skewed columns
df['views'] = np.log1p(df['views'])
df['comments'] = np.log1p(df['comments'])

# Features and target
tfidf_cols = [c for c in df.columns if c.startswith("title_") or c.startswith("firsttag_")]
category_cols = [c for c in df.columns if c.startswith("category_")]

X = df[['uploaddate','comments','title_length','title_word_count','first_tag_length'] + tfidf_cols + category_cols]
y = df['views']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# Train Models
# ----------------------------
models = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42, eval_metric='rmse')
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    print(f"--- {name} ---")
    print("Test MSE:", mean_squared_error(y_test, y_pred))
    print("Test R²:", r2_score(y_test, y_pred))

    # Feature importance
    if name == "RandomForest":
        importances = model.feature_importances_
        feature_names = X.columns
        fi_df = pd.DataFrame({'Feature':feature_names, 'Importance':importances}).sort_values(by='Importance', ascending=False)
        plt.figure(figsize=(10,8))
        sns.barplot(x='Importance', y='Feature', data=fi_df.head(20))
        plt.title(f"{name} Feature Importance (Scraped Model)")
        plt.tight_layout()
        plt.savefig(f"{name}_scraped_feature_importance.png")
        plt.close()
    else:
        # For XGBoost
        from xgboost import plot_importance
        plt.figure(figsize=(10,8))
        plot_importance(model, max_num_features=20)
        plt.title(f"{name} Feature Importance (Scraped Model)")
        plt.tight_layout()
        plt.savefig(f"{name}_scraped_feature_importance.png")
        plt.close()

    # Actual vs Predicted
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Actual Views (log)")
    plt.ylabel("Predicted Views (log)")
    plt.title(f"{name} Scraped Model: Actual vs Predicted")
    plt.savefig(f"{name}_scraped_actual_vs_pred.png")
    plt.close()
