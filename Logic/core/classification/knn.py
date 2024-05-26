import numpy as np
from scipy.spatial.distance import euclidean
from scipy.stats import mode
from sklearn.metrics import classification_report
from tqdm import tqdm

from Logic.core.classification.basic_classifier import BasicClassifier
from Logic.core.classification.data_loader import ReviewLoader


class KnnClassifier(BasicClassifier):
    def __init__(self, n_neighbors):
        super().__init__()
        self.k = n_neighbors

    def fit(self, X, y):
        """
        Fit the model using X as training data and y as target values
        use the Euclidean distance to find the k nearest neighbors
        Warning: Maybe you need to reduce the size of X to avoid memory errors

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
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, x):
        """
        Parameters
        ----------
        x: np.ndarray
            An k * n matrix - k is count of docs and n is embedding size
        Returns
        -------
        np.ndarray
            Return the predicted class for each doc
            with the highest probability (argmax)
        """
        predictions = np.zeros(len(x))
        for i, sample in enumerate(tqdm(x)):
            distances = [euclidean(sample, self.X_train[j]) for j in range(len(self.X_train))]
            indices = np.argsort(distances)[:self.k]
            labels = self.y_train[indices]
            predictions[i] = mode(labels).mode
        return predictions

    def prediction_report(self, x, y):
        """
        Parameters
        ----------
        x: np.ndarray
            An k * n matrix - k is count of docs and n is embedding size
        y: np.ndarray
            The real class label for each doc
        Returns
        -------
        str
            Return the classification report
        """
        y_pred = self.predict(x)
        return classification_report(y, y_pred, zero_division=1)


# F1 Accuracy : 70%
if __name__ == '__main__':
    """
    Fit the model with the training data and predict the test data, then print the classification report
    """
    review_loader = ReviewLoader(file_path='training_data/IMDB_Dataset.csv')
    # review_loader = ReviewLoader(file_path='mini-dataset')
    review_loader.load_data()
    X_train, X_test, y_train, y_test = review_loader.split_data(test_data_ratio=0.2)

    knn_classifier = KnnClassifier(n_neighbors=10)
    knn_classifier.fit(X_train, y_train)

    print(knn_classifier.prediction_report(X_test, y_test))
