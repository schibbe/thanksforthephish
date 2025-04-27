import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import FunctionTransformer
from scipy.sparse import hstack

def load_data_from_folder(folder_path):
    all_data = []

    for filename in os.listdir(folder_path):
        if not filename.endswith(".csv"):
            continue

        file_path = os.path.join(folder_path, filename)
        try:
            df = pd.read_csv(file_path, encoding="utf-8")
        except UnicodeDecodeError:
            print(f"UTF-8 failed with {filename}, trying ISO-8859-1...")
            try:
                df = pd.read_csv(file_path, encoding="ISO-8859-1")
            except Exception as e:
                print(f"Error when reading {filename}: {e}")
                continue

        print(f"File {filename} - Columns: {list(df.columns)}")

        if {"subject", "body", "label"}.issubset(df.columns):
            df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")
        elif {"Email_Subject", "Email_Content"}.issubset(df.columns):
            df["text"] = df["Email_Subject"].fillna("") + " " + df["Email_Content"].fillna("")
            df["label"] = 1
        elif {"subject", "label"}.issubset(df.columns):
            df["text"] = df["subject"].fillna("")
        elif {"sender", "label"}.issubset(df.columns):
            df["text"] = df["sender"].fillna("")
        elif {"text", "label"}.issubset(df.columns):
            pass
        else:
            print(f"File {filename} has no matching columns. Skipped.")
            continue

        all_data.append(df[["text", "label"]])

    if not all_data:
        raise ValueError("No valid css files found.")

    return pd.concat(all_data, ignore_index=True)

def count_urls(text):
    if isinstance(text, str):
        return len(re.findall(r'http[s]?://\S+', text))
    else:
        return 0

def add_url_count_feature(df):
    return df["text"].apply(count_urls).to_frame()

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Ham", "Phishing"], yticklabels=["Ham", "Phishing"])
    plt.title("Confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

def plot_bar_chart(y):
    counts = y.value_counts()
    counts.index = ["Ham" if i == 0 else "Phishing" for i in counts.index]
    sns.barplot(x=counts.index, y=counts.values)
    plt.title("Distribution of classes")
    plt.ylabel("Quantity")
    plt.xlabel("Class")
    plt.show()

def plot_roc_curve(y_test, y_proba):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

def show_top_words(vectorizer, model, n=15):
    try:
        feature_names = vectorizer.get_feature_names_out()
        log_probs = model.feature_log_prob_[1][:len(feature_names)]  
        top_phishing = log_probs.argsort()[-n:]

        print("\Top-Words for Phishing-Mails:")
        for idx in reversed(top_phishing):
            print(f"- {feature_names[idx]}")
    except Exception as e:
        print(f"Error: {e}")

def train_and_evaluate(df):
    X = df["text"]
    y = df["label"]

    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    X_vect = vectorizer.fit_transform(X)

    url_count = add_url_count_feature(df)
    X_combined = hstack([X_vect, url_count])

    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n=== Classification report ===")
    print(classification_report(y_test, y_pred))

    plot_bar_chart(y)
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_proba)
    show_top_words(vectorizer, model)

    os.makedirs("models", exist_ok=True)
    
    joblib.dump(model, "models/phishing_model.joblib")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")
    joblib.dump(url_count, "models/url_count_transformer.joblib")
    print("Model and Vectorizer saved.")

def main():
    folder_path = "data"
    df = load_data_from_folder(folder_path)
    train_and_evaluate(df)

if __name__ == "__main__":
    main()
