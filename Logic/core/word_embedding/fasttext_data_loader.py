import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

__all__ = ['FastTextDataLoader']

class FastTextDataLoader:
    """
    This class is designed to load and pre-process data for training a FastText model.

    It takes the file path to a data source containing movie information (synopses, summaries, reviews, titles, genres) as input.
    The class provides methods to read the data into a pandas DataFrame, pre-process the text data, and create training data (features and labels)
    """

    def __init__(self, file_path, preprocessor=None):
        """
        Initializes the FastTextDataLoader class with the file path to the data source.

        Parameters
        ----------
        file_path: str
            The path to the file containing movie information.
        """
        self.file_path = file_path
        if preprocessor is None:
            preprocessor = lambda x: x
        self.preprocessor = preprocessor

    def read_data_to_df(self):
        """
        Reads data from the specified file path and creates a pandas DataFrame containing movie information.

        You can use an IndexReader class to access the data based on document IDs.
        It extracts synopses, summaries, reviews, titles, and genres for each movie.
        The extracted data is then stored in a pandas DataFrame with appropriate column names.

        Returns
        ----------
            pd.DataFrame: A pandas DataFrame containing movie information (synopses, summaries, reviews, titles, genres).
        """
        if self.file_path.endswith('.json'):
            return pd.read_json(self.file_path)
        return pd.read_csv(self.file_path)

    def _get_data_from_df(self, df: pd.DataFrame, colomn: str):
        new_df = df[[colomn, 'genres']].dropna()
        new_df[colomn] = new_df[colomn].apply(self.preprocessor)
        return zip(*new_df.to_numpy())

    def _get_data_and_explode_from_df(self, df: pd.DataFrame, colomn: str):
        new_df = df[[colomn, 'genres']].explode(colomn).dropna().reset_index(drop=True)
        new_df[colomn] = new_df[colomn].apply(lambda x: self.preprocessor(x[0]))
        return zip(*new_df.to_numpy())

    def create_train_data(self):
        """
        Reads data using the read_data_to_df function, pre-processes the text data, and creates training data (features and labels).

        Returns:
            tuple: A tuple containing two NumPy arrays: X (preprocessed text data) and y (encoded genre labels).
        """
        df = self.read_data_to_df()
        multi_label_binarizer = MultiLabelBinarizer()
        multi_label_binarizer.fit(df['genres'])
        
        X_synposis, y_synposis = self._get_data_and_explode_from_df(df, 'synposis')
        X_summaries, y_summaries = self._get_data_and_explode_from_df(df, 'summaries')
        X_reviews, y_reviews = self._get_data_and_explode_from_df(df, 'reviews')
        X_title, y_title = self._get_data_from_df(df, 'title')
        
        X = X_synposis + X_summaries + X_reviews + X_title
        y = multi_label_binarizer.transform(y_synposis + y_summaries + y_reviews + y_title)
        
        return X, y
