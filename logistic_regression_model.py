import time
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class LogisticRegressionSentimentModel:
    # TF-IDF + Logistic Regression baseline model.
    # This model treats text as a bag of weighted word features rather than an ordered sequence

    def __init__(self, max_features=20000, max_iter=1000, random_state=42):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words=None
        )

        self.model = LogisticRegression(
            max_iter=max_iter,
            random_state=random_state
        )

        self.training_time_seconds = None
        self.inference_time_seconds = None

    def fit(self, train_df):
        # Fit TF-IDF and Logistic Regression on the training DataFrame
        start_time = time.time()

        x_train = self.vectorizer.fit_transform(train_df["text"])
        y_train = train_df["label"].values

        self.model.fit(x_train, y_train)

        end_time = time.time()
        self.training_time_seconds = end_time - start_time

    def predict(self, df):
        # Predict hard labels for a DataFrame with a text column
        start_time = time.time()

        x = self.vectorizer.transform(df["text"])
        predictions = self.model.predict(x)

        end_time = time.time()
        self.inference_time_seconds = end_time - start_time

        return predictions

    def predict_proba(self, df):
        # Predict positive-class probabilities for AUC computation
        x = self.vectorizer.transform(df["text"])
        probabilities = self.model.predict_proba(x)[:, 1]
        return probabilities
