import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import shap

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

csv_path = "/Users/andrewherman/CMP 455/Project 1/Scraped/Youtube_Video_Data.csv"
df = pd.read_csv(csv_path)
cols = [col.lower() for col in df.columns]


def get_column(kw):
    for i, col in enumerate(cols):
        if kw.lower() == col:
            return df.columns[i]
    return None


video_title_col = get_column("title")
url_col = get_column("url")
views_col = get_column("views")
upload_date = get_column("uploaddate")
comments_col = get_column("comments")
first_tag_col = get_column("firsttag")
category_col = get_column("category")


# Remove "views" and value symbol
def convert_views(view_str):
    if pd.isnull(view_str):  # Missing views
        return 0

    view_str = view_str.lower().replace("views", "").strip()  # Remove view string and remove spaces
    if "k" in view_str: return int(float(view_str.replace("k", "")) * 1000)  # Convert k to 1000 views
    if 'm' in view_str: return int(float(view_str.replace("m", "")) * 1_000_000)  # Convert m to a million views
    try:
        return int(float(view_str))
    except:
        return 0


# Convert string to days
def convert_date(date_str):
    if pd.isnull(date_str): return 0
    date = date_str.split()
    try:
        val = int(date[0])
        unit = date[1]
        if unit in ["year", "years"]:
            return val * 365
        elif unit in ["month", "months"]:
            return val * 30
        elif unit in ["week", "weeks"]:
            return val * 7
        elif unit in ["day", "days"]:
            return val
        else:
            return 0
    except:
        return 0


def convert_comments(comment_str):
    if pd.isnull(comment_str):
        return 0
    comment_str = comment_str.replace("Comments", "").replace(",", "").strip()
    try:
        return int(float(comment_str))
    except:
        return 0


def convert_tag_and_title(df, column):
    # Convert text into numbers and remove filler words
    vectorizer = TfidfVectorizer(stop_words="english", max_features=25)
    tfidf = vectorizer.fit_transform(df[column].fillna(""))  # For missing values
    names = [f"{column.lower()}_{w.lower()}" for w in vectorizer.get_feature_names_out()]
    new_df = pd.DataFrame(tfidf.toarray(), columns=names, index=df.index).round(2)
    return pd.concat([df, new_df], axis=1)


df["Views"] = df["Views"].apply(convert_views)  # Convert Views
df["UploadDate"] = df["UploadDate"].apply(convert_date)
df["Comments"] = df["Comments"].apply(convert_comments)
df = convert_tag_and_title(df, "Title")
df = convert_tag_and_title(df, "FirstTag")

# Rename headers
df.columns = df.columns.str.lower()

df = pd.get_dummies(df, columns=["category"], drop_first=True)  # One-Hot encoder for category

"""
Preprocessing
"""
# Remove rows with missing target value
df = df.dropna(subset=["views"])  # Remove missing values

# Fill in missing values with median
for col in ["uploaddate", "comments"]:
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
for col in ["views", "comments"]:
    df[col] = np.log1p(df[col])

# Convert title and first tag into meaningful numerical values
df["title_length"] = df["title"].apply(lambda title: len(str(title)))
df["title_word_count"] = df["title"].apply(lambda title: len(str(title).split()))
df["first_tag_length"] = df["firsttag"].apply(lambda firsttag: len(str(firsttag)))

# Train test split data
TFIDF_titles = [col for col in df.columns if col.startswith("title_")]
TFIDF_tags = [col for col in df.columns if col.startswith("firsttag_")]
category_dummies = [col for col in df.columns if col.startswith("category_")]

X = df[["uploaddate", "comments", "title_length", "title_word_count",
        "first_tag_length"] + TFIDF_titles + TFIDF_tags + category_dummies]
y = df["views"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to dataframe
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

model = LinearRegression()
selector = RFE(estimator=model, n_features_to_select=25)
selector.fit(X_train_scaled_df, y_train)

# Get selected feature names
selected_features = X_train_scaled_df.columns[selector.support_]
print(selected_features)

# Only use most important features to train model
X_train_selected = X_train_scaled_df[selected_features]
X_test_selected = X_test_scaled_df[selected_features]

regressor = LinearRegression()
regressor.fit(X_train_selected, y_train)

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

explainer = shap.Explainer(regressor, X_test_selected)
shap_values = explainer(X_test_selected)

# Summary plot (global feature importance)
shap.summary_plot(shap_values, X_test_selected)

# Actual vs Predicted
plt.scatter(y_test, y_pred_test, alpha=0.5)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs. Predicted Views")
plt.show()

# Why the selected features make the model more accurate
ranking = pd.Series(selector.ranking_, index=X_train.columns)
bar_color = ["green" if r == 1 else "red" for r in ranking]
plt.figure(figsize=(10, 12))
ranking.sort_values(ascending=False).plot(kind='barh', color=bar_color)
plt.xlabel("RFE Ranking")
plt.ylabel("Feature")
plt.title("Selected vs Non-selected Features")
plt.gca().invert_yaxis()
plt.show()
