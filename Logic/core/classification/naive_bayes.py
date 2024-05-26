import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from Logic.core.classification.basic_classifier import BasicClassifier
from Logic.core.classification.data_loader import ReviewLoader


class NaiveBayes(BasicClassifier):
    def __init__(self, count_vectorizer, alpha=1):
        super().__init__()
        self.cv = count_vectorizer
        self.num_classes = None
        self.classes = None
        self.number_of_features = None
        self.number_of_samples = None
        self.prior = None
        self.feature_probabilities = None
        self.log_probs = None
        self.log_prior = None
        self.alpha = alpha

    def fit(self, X, y):
        """
        Fit the features and the labels
        Calculate prior and feature probabilities

        Parameters
        ----------
        X: np.ndarray
            An m * n matrix - m is count of docs and n is embedding size

        y: np.ndarray
            The real class label for each doc

        Returns
        -------
        self
            Returns self as a classifier
        """
        self.number_of_samples, self.number_of_features = X.shape
        self.classes = np.unique(y)
        self.num_classes = len(self.classes)
        self.prior = np.bincount(y) / self.number_of_samples
        self.feature_probabilities = np.zeros((self.num_classes, self.number_of_features))
        for i, label in enumerate(self.classes):
            labels = X[y == label]
            counts = np.sum(labels, axis=0)
            self.feature_probabilities[i, :] = (counts + self.alpha) / (np.sum(counts) + self.alpha*self.number_of_features)
        self.log_probs = np.log(self.feature_probabilities)
        self.log_prior = np.log(self.prior)
        return self

    def predict(self, X):
        """
        Parameters
        ----------
        X: np.ndarray
            An k * n matrix - k is count of docs and n is embedding size
        Returns
        -------
        np.ndarray
            Return the predicted class for each doc
            with the highest probability (argmax)
        """
        scores = self.log_prior + (X @ self.log_probs.T)
        return self.classes[np.argmax(scores, axis=1)]

    def prediction_report(self, X, y):
        """
        Parameters
        ----------
        X: np.ndarray
            An k * n matrix - k is count of docs and n is embedding size
        y: np.ndarray
            The real class label for each doc
        Returns
        -------
        str
            Return the classification report
        """
        y_pred = self.predict(X)
        return classification_report(y, y_pred)

    def get_percent_of_positive_reviews(self, sentences):
        """
        You have to override this method because we are using a different embedding method in this class.
        """
        loader = ReviewLoader('training_data/IMDB_Dataset.csv')
        loader.load_data()
        positive_index = loader.sentiments.index('positive')
        loader.sentiments = LabelEncoder().fit_transform(loader.sentiments)
        predictions = self.predict(self.cv.transform(sentences).toarray())
        return 100 * (np.sum(predictions == loader.sentiments[positive_index]) / len(sentences))


# F1 Accuracy : 85%
if __name__ == '__main__':
    """
    First, find the embeddings of the reviews using the CountVectorizer, then fit the model with the training data.
    Finally, predict the test data and print the classification report
    You can use scikit-learn's CountVectorizer to find the embeddings.
    """
    review_loader = ReviewLoader('training_data/IMDB_Dataset.csv')
    review_loader.load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        review_loader.review_tokens,
        np.array(review_loader.sentiments),
        test_size=0.2
    )

    count_vectorizer = CountVectorizer()
    X_train = count_vectorizer.fit_transform(X_train)
    X_test = count_vectorizer.transform(X_test)

    nb_classifier = NaiveBayes(count_vectorizer, alpha=1)
    nb_classifier.fit(X_train, y_train)

    print(nb_classifier.prediction_report(X_test, y_test))