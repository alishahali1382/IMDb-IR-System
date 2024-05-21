import fasttext
import re
import string
import numpy as np

from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from scipy.spatial import distance

from .fasttext_data_loader import FastTextDataLoader

__all__ = ['FastText']

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def preprocess_text(text: str, minimum_length=1, stopword_removal=True, stopwords_domain=[], lower_case=True,
                       punctuation_removal=True):
    """
    preprocess text by removing stopwords, punctuations, and converting to lowercase, and also filter based on a min length
    for stopwords use nltk.corpus.stopwords.words('english')
    for punctuations use string.punctuation

    Parameters
    ----------
    text: str
        text to be preprocessed
    minimum_length: int
        minimum length of the token
    stopword_removal: bool
        whether to remove stopwords
    stopwords_domain: list
        list of stopwords to be removed base on domain
    lower_case: bool
        whether to convert to lowercase
    punctuation_removal: bool
        whether to remove punctuations
    """
    if lower_case:
        text = text.lower()

    if punctuation_removal:
        text = text.translate(str.maketrans('', '', string.punctuation))

    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(stemmer.stem(word)) for word in tokens]

    if stopword_removal:
        stop_words = set(stopwords.words('english')).union(set(stopwords_domain))
        tokens = [word for word in tokens if word not in stop_words]

    tokens = [word for word in tokens if len(word) >= minimum_length]

    return ' '.join(tokens)

class FastText:
    """
    A class used to train a FastText model and generate embeddings for text data.

    Attributes
    ----------
    method : str
        The training method for the FastText model.
    model : fasttext.FastText._FastText
        The trained FastText model.
    """

    def __init__(self, preprocessor, method='skipgram'):
        """
        Initializes the FastText with a preprocessor and a training method.

        Parameters
        ----------
        method : str, optional
            The training method for the FastText model.
        """
        self.preprocessor = preprocessor
        self.method = method
        self.model = None


    def train(self, texts):
        """
        Trains the FastText model with the given texts.

        Parameters
        ----------
        texts : list of str
            The texts to train the FastText model.
        """
        with open("fasttext_train.txt", "w") as f:
            for text in tqdm(texts):
                f.write(self.preprocessor(text) + "\n")

        self.model = fasttext.train_unsupervised("fasttext_train.txt", model=self.method)

    def get_query_embedding(self, query):
        """
        Generates an embedding for the given query.

        Parameters
        ----------
        query : str
            The query to generate an embedding for.
        tf_idf_vectorizer : sklearn.feature_extraction.text.TfidfVectorizer
            The TfidfVectorizer to transform the query.
        do_preprocess : bool, optional
            Whether to preprocess the query.

        Returns
        -------
        np.ndarray
            The embedding for the query.
        """
        query = self.preprocessor(query)
        return self.model.get_sentence_vector(query)

    def analogy(self, word1, word2, word3):
        """
        Perform an analogy task: word1 is to word2 as word3 is to __.

        Args:
            word1 (str): The first word in the analogy.
            word2 (str): The second word in the analogy.
            word3 (str): The third word in the analogy.

        Returns:
            str: The word that completes the analogy.
        """
        word1 = self.preprocessor(word1)
        word2 = self.preprocessor(word2)
        word3 = self.preprocessor(word3)

        # Obtain word embeddings for the words in the analogy
        vec1 = self.model.get_word_vector(word1)
        vec2 = self.model.get_word_vector(word2)
        vec3 = self.model.get_word_vector(word3)

        # Perform vector arithmetic
        vec4 = vec2 - vec1 + vec3

        # Create a dictionary mapping each word in the vocabulary to its corresponding vector
        all_words = self.model.get_words()
        word_to_vec = {word: self.model.get_word_vector(word) for word in all_words}

        # Exclude the input words from the possible results
        word_to_vec.pop(word1)
        word_to_vec.pop(word2)
        word_to_vec.pop(word3)

        # Find the word whose vector is closest to the result vector
        min_distance = float('inf')
        best_word = None
        for word, vec in word_to_vec.items():
            dist = distance.euclidean(vec, vec4)
            if dist < min_distance:
                min_distance = dist
                best_word = word
        print(f"Best word: {best_word}, Distance: {min_distance}")
        return best_word

    def save_model(self, path='FastText_model.bin'):
        """
        Saves the FastText model to a file.

        Parameters
        ----------
        path : str, optional
            The path to save the FastText model.
        """
        self.model.save_model(path)

    def load_model(self, path="FastText_model.bin"):
        """
        Loads the FastText model from a file.

        Parameters
        ----------
        path : str, optional
            The path to load the FastText model.
        """
        self.model = fasttext.load_model(path)

    def prepare(self, dataset, mode, save=False, path='FastText_model.bin'):
        """
        Prepares the FastText model.

        Parameters
        ----------
        dataset : list of str
            The dataset to train the FastText model.
        mode : str
            The mode to prepare the FastText model.
        """
        if mode == 'train':
            self.train(dataset)
        elif mode == 'load':
            self.load_model(path)
        if save:
            self.save_model(path)

if __name__ == "__main__":
    import nltk
    nltk.download('stopwords')
    
    ft_model = FastText(preprocessor=preprocess_text, method='skipgram')
    # ft_model = FastText(method='skipgram')

    path = './training_data/IMDB_crawled.json'
    ft_data_loader = FastTextDataLoader(path)

    X, _ = ft_data_loader.create_train_data()

    # ft_model.prepare(X, mode = "train", save = True)
    ft_model.prepare(X, mode = "load")

    print(10 * "*" + "Similarity" + 10 * "*")
    word = 'queen'
    neighbors = ft_model.model.get_nearest_neighbors(word, k=5)

    for neighbor in neighbors:
        print(f"Word: {neighbor[1]}, Similarity: {neighbor[0]}")

    print(10 * "*" + "Analogy" + 10 * "*")
    word1 = "man"
    word2 = "woman"
    word3 = "boy"
    print(f"Similarity between {word1} and {word2} is like similarity between {word3} and {ft_model.analogy(word1, word2, word3)}")
