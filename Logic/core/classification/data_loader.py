import numpy as np
import pandas as pd
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ..word_embedding.fasttext_model import FastText, preprocess_text


class ReviewLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.fasttext_model = None
        self.review_tokens = []
        self.sentiments = []
        self.embeddings = []

    def load_data(self):
        """
        Load the data from the csv file and preprocess the text. Then save the normalized tokens and the sentiment labels.
        Also, load the fasttext model.
        """
        self.fasttext_model = FastText(preprocess_text)
        self.fasttext_model.load_model()
        reviews_data = pd.read_csv(self.file_path)
        self.review_tokens = reviews_data['review'].apply(preprocess_text).tolist()
        label_encoder = LabelEncoder()
        self.sentiments = label_encoder.fit_transform(reviews_data['sentiment'])
        print("Successfully loaded and preprocessed reviews data")

    def get_embeddings(self):
        """
        Get the embeddings for the reviews using the fasttext model.
        """
        self.embeddings = np.array([
            self.fasttext_model.get_query_embedding(review)
            for review in tqdm.tqdm(self.review_tokens)
        ])

    def split_data(self, test_data_ratio=0.2):
        """
        Split the data into training and testing data.

        Parameters
        ----------
        test_data_ratio: float
            The ratio of the test data
        Returns
        -------
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
            Return the training and testing data for the embeddings and the sentiments.
            in the order of x_train, x_test, y_train, y_test
        """
        self.get_embeddings()
        X_train, X_test, y_train, y_test = train_test_split(self.embeddings, self.sentiments, test_size=test_data_ratio)
        return X_train, X_test, y_train, y_test
