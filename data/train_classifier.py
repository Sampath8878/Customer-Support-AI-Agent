import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

# 1. Load dataset
df = pd.read_csv("tickets.csv")

# 2. Combine subject + body as the input text
df["text"] = (df["subject"].fillna("") + " " + df["body"].fillna("")).str.strip()

# 3. Drop rows with missing values
df = df.dropna(subset=["text", "label"])

# 4. Define features and target
X = df["text"]
y = df["label"]

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 6. Build pipeline (TF-IDF + Logistic Regression)
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1,2))),  # bi-grams help separate delivery/defect
    ("clf", LogisticRegression(max_iter=200, class_weight="balanced"))
])

# 7. Train model
pipeline.fit(X_train, y_train)

# 8. Evaluate
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# 9. Save model
joblib.dump(pipeline, "ticket_classifier.pkl")
print("✅ Model saved as ticket_classifier.pkl")
