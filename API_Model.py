import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

csv_path = "/Users/jto5108/Project1/CMPSC455/Video_Metadata.csv"
df = pd.read_csv(csv_path)
cols = [col.lower() for col in df.columns]


def get_column(kw):
    for i, col in enumerate(cols):
        if kw.lower() == col:
            return df.columns[i]
    return None


video_title_col = get_column("title")
duration_col = get_column("duration_seconds")
days_since_col = get_column("days_since_upload")
views_col = get_column("views")
likes_col = get_column("likes")
comments_col = get_column("comments")
channel_title_col = get_column("channel_title")
subscribers_col = get_column("subscribers")

# print("max", max(df["views"]))
# print("min", min(df["views"]))
# print("Mean", df["views"].mean())
# print("Median", df["views"].median())
# print("Std", df["views"].std())

# df = df[video_title_col, duration_col, days_since_col, views_col, likes_col, comments_col, channel_title_col, subscribers_col]
df.columns = ["id", "video_title", "duration", "days_since", "views", "likes", "comments", "channel_title",
              "subscribers"]
"""
Preprocessing
"""
# Remove rows with missing target value
df = df.dropna(subset=["views"])  # Remove missing values

# Fill in missing values with median
for col in ["likes", "subscribers", "duration", "days_since", "comments"]:
    df[col] = df[col].fillna(df[col].median())

# plt.hist(df["views"], bins=50, log=True)
# plt.xlabel("Views")
# plt.ylabel("Frequency")
# plt.title("Histogram of Views")
# plt.show()

# Prevent extreme outliers
lower, upper = df["views"].quantile([0.02, 0.98])
df["views"] = np.clip(df["views"], lower, upper)

# Log tranform for skewed numerical values
for col in ["views", "likes", "comments", "subscribers"]:
    df[col] = np.log1p(df[col])

# Convert title and channel name into meaningful numerical values
df["title_length"] = df["video_title"].apply(lambda title: len(title))
df["title_word_count"] = df["video_title"].apply(lambda title: len(str(title).split()))
df["channel_name_length"] = df["channel_title"].apply(lambda channel_name: len(channel_name))

# Train test split data
X = df[["duration", "days_since", "likes", "comments", "subscribers", "title_length", "title_word_count",
        "channel_name_length"]]
y = df["views"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 1

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# Convert back to dataframe
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

# Apply Recursive Feature Elimination to get 5 most meaninful features
model = LinearRegression()
selector = RFE(estimator=model, n_features_to_select=5)
selector.fit(X_train_scaled_df, y_train)

# Get selected feature names
selected_features = X_train.columns[selector.support_]
print("Selected features:", list(selected_features))

# Only use most important features to train model
X_train_selected = X_train_scaled_df[selected_features]
X_test_selected = X_test_scaled_df[selected_features]

# Create the linear regression model
regressor = LinearRegression()
regressor.fit(X_train_selected, y_train)

# Make predictions
y_pred_train = regressor.predict(X_train_selected)
y_pred_test = regressor.predict(X_test_selected)

print("Training MSE:", mean_squared_error(y_train, y_pred_train))
print("Training R-Squared:", r2_score(y_train, y_pred_train))

print("Test MSE:", mean_squared_error(y_test, y_pred_test))
print("Test R-Squared:", r2_score(y_test, y_pred_test))
print("Score: ", regressor.score(X_train_selected, y_train))
"""
Visualization
"""
# Actual vs Predicted
plt.scatter(y_test, y_pred_test, alpha=0.5)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs. Predicted Views")
plt.show()

# Why the selected features make the model more accurate
ranking = pd.Series(selector.ranking_, index=X_train.columns)
ranking = ranking.sort_values()
plt.figure(figsize=(15, 6))
bar_color = ["green" if r == 1 else "red" for r in ranking]
plt.bar(ranking.index, ranking.values, width=0.5, color=bar_color)
plt.xlabel("Features")
plt.ylabel("RFE Ranking (1 = selected)")
plt.title("Feature Importance by RFE")
plt.show()

# Engagment Trends
for feature in ["likes", "subscribers", "duration", "comments"]:
    sns.scatterplot(x=feature, y="views", data=df)
    plt.xlabel(f"{feature.upper()}")
    plt.ylabel("Views")
    plt.title(f"Views vs {feature.upper()}")
    plt.show()
