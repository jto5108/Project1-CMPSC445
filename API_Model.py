import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE

# ----------------------------
# Load Data
# ----------------------------
csv_path = "youtube_videos.csv"
df = pd.read_csv(csv_path)

# Rename columns for simplicity
df.columns = ["id", "video_title", "duration", "days_since", "views", "likes",
              "comments", "channel_title", "subscribers"]

# ----------------------------
# Preprocessing
# ----------------------------
# Drop rows with missing target
df = df.dropna(subset=['views'])

# Fill NaNs in numeric columns
for col in ['likes', 'subscribers', 'duration', 'days_since', 'comments']:
    df[col] = df[col].fillna(df[col].median())

# Clip extreme outliers
lower, upper = df['views'].quantile([0.02, 0.98])
df['views'] = np.clip(df['views'], lower, upper)

# Log-transform skewed columns
for col in ['views', 'likes', 'comments', 'subscribers']:
    df[col] = np.log1p(df[col])

# Feature engineering
df['title_length'] = df['video_title'].apply(lambda x: len(str(x)))
df['title_word_count'] = df['video_title'].apply(lambda x: len(str(x).split()))
df['channel_name_length'] = df['channel_title'].apply(lambda x: len(str(x)))

# ----------------------------
# Train/Test Split
# ----------------------------
X = df[['duration', 'days_since', 'likes', 'comments', 'subscribers',
        'title_length', 'title_word_count', 'channel_name_length']]
y = df['views']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# Recursive Feature Elimination
# ----------------------------
model = LinearRegression()
selector = RFE(estimator=model, n_features_to_select=5)
selector.fit(X_train_scaled, y_train)

selected_features = X.columns[selector.support_]
print("Selected features:", list(selected_features))

X_train_selected = pd.DataFrame(X_train_scaled, columns=X.columns)[selected_features]
X_test_selected = pd.DataFrame(X_test_scaled, columns=X.columns)[selected_features]

# ----------------------------
# Model Training: Random Forest
# ----------------------------
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train_selected, y_train)

y_pred_train_rf = rf_model.predict(X_train_selected)
y_pred_test_rf = rf_model.predict(X_test_selected)

print("\nRandom Forest Performance:")
print("Train MSE:", mean_squared_error(y_train, y_pred_train_rf))
print("Train R2:", r2_score(y_train, y_pred_train_rf))
print("Test MSE:", mean_squared_error(y_test, y_pred_test_rf))
print("Test R2:", r2_score(y_test, y_pred_test_rf))

# Feature Importance
plt.figure(figsize=(10,6))
feat_imp_rf = pd.Series(rf_model.feature_importances_, index=X_train_selected.columns)
feat_imp_rf.sort_values(ascending=True).plot(kind='barh', color='skyblue')
plt.title("Random Forest Feature Importance")
plt.savefig("rf_feature_importance.png")
plt.close()

# Actual vs Predicted Plot
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_test_rf, alpha=0.5)
plt.xlabel("Actual Views")
plt.ylabel("Predicted Views")
plt.title("Random Forest: Actual vs Predicted Views")
plt.savefig("rf_actual_vs_pred.png")
plt.close()

# ----------------------------
# Model Training: XGBoost
# ----------------------------
xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train_selected, y_train)

y_pred_train_xgb = xgb_model.predict(X_train_selected)
y_pred_test_xgb = xgb_model.predict(X_test_selected)

print("\nXGBoost Performance:")
print("Train MSE:", mean_squared_error(y_train, y_pred_train_xgb))
print("Train R2:", r2_score(y_train, y_pred_train_xgb))
print("Test MSE:", mean_squared_error(y_test, y_pred_test_xgb))
print("Test R2:", r2_score(y_test, y_pred_test_xgb))

# Feature Importance
plt.figure(figsize=(10,6))
feat_imp_xgb = pd.Series(xgb_model.feature_importances_, index=X_train_selected.columns)
feat_imp_xgb.sort_values(ascending=True).plot(kind='barh', color='orange')
plt.title("XGBoost Feature Importance")
plt.savefig("xgb_feature_importance.png")
plt.close()

# Actual vs Predicted Plot
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_test_xgb, alpha=0.5)
plt.xlabel("Actual Views")
plt.ylabel("Predicted Views")
plt.title("XGBoost: Actual vs Predicted Views")
plt.savefig("xgb_actual_vs_pred.png")
plt.close()

# ----------------------------
# Engagement Trends
# ----------------------------
for feature in ['duration', 'comments', 'likes', 'subscribers']:
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=feature, y='views', data=df)
    plt.xlabel(feature.capitalize())
    plt.ylabel("Views")
    plt.title(f"Views vs {feature.capitalize()}")
    plt.savefig(f"engagement_{feature}.png")
    plt.close()
