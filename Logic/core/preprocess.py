import re
from typing import Dict, List
import nltk

def _download_nltk():
    nltk.download('wordnet')
    nltk.download('punkt')

class Preprocessor:

    def __init__(self):
        """
        Initialize the class.
        """
        stopwords_path = "/".join(__file__.split('/')[:-1]+['stopwords.txt'])
        with open(stopwords_path, 'r') as f:
            self.stopwords = set(f.read().split('\n'))

        self.stemmer = nltk.stem.PorterStemmer()
        self.lemmatizer = nltk.stem.WordNetLemmatizer()

    def preprocess_movie(self, movie: Dict[str, str]) -> Dict[str, List[str]]:
        """
        Preprocess the movie data.

        Parameters
        ----------
        movie : Dict[str, str]
            The movie data

        Returns
        ----------
        Dict[str, List[str]]
            The preprocessed movie data
        """
        movie["stars"] = self.name_preprocess(movie["stars"])
        movie["genres"] = self.name_preprocess(movie["genres"])
        movie["summaries"] = self.preprocess(movie["summaries"])
        return movie

    def name_preprocess(self, document: str) -> List[str]:
        """
        Preprocess the text of names, without normalizing, removing stopwords.

        Parameters
        ----------
        documents : str
            The document to be preprocessed

        Returns
        ----------
        List[str]
            The preprocessed documents.
        """
        document = document.lower()
        document = self.remove_links(document)
        document = self.remove_punctuations(document)
        words = self.tokenize(document)
        return words        

    def preprocess(self, document: str) -> List[str]:
        """
        Preprocess the text using the methods in the class.

        Parameters
        ----------
        documents : str
            The document to be preprocessed

        Returns
        ----------
        List[str]
            The preprocessed documents.
        """
        document = self.normalize(document)
        document = self.remove_links(document)
        document = self.remove_punctuations(document)
        words = self.tokenize(document)
        words = self.remove_stopwords(words)
        return words

    def normalize(self, text: str) -> str:
        """
        Normalize the text by converting it to a lower case, stemming, lemmatization, etc.

        Parameters
        ----------
        text : str
            The text to be normalized.

        Returns
        ----------
        str
            The normalized text.
        """
        text = self.stemmer.stem(text, to_lowercase=True)
        text = self.lemmatizer.lemmatize(text)
        return text

    def remove_links(self, text: str) -> str:
        """
        Remove links from the text.

        Parameters
        ----------
        text : str
            The text to be processed.

        Returns
        ----------
        str
            The text with links removed.
        """
        # NOTE: I almost did this on crawling part.
        text = re.sub(r'http[^\s]+', '', text)
        return text

    def remove_punctuations(self, text: str) -> str:
        """
        Remove punctuations from the text.

        Parameters
        ----------
        text : str
            The text to be processed.

        Returns
        ----------
        str
            The text with punctuations removed.
        """
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize the words in the text.

        Parameters
        ----------
        text : str
            The text to be tokenized.

        Returns
        ----------
        list
            The list of words.
        """
        return nltk.word_tokenize(text)

    def remove_stopwords(self, words: List[str]) -> List[str]:
        """
        Remove stopwords from the text.

        Parameters
        ----------
        text : str
            The text to remove stopwords from.

        Returns
        ----------
        list
            The list of words with stopwords removed.
        """
        return [word for word in words if word not in self.stopwords]

if __name__ == "__main__":
    _download_nltk()
    preprocessor = Preprocessor()
    print(preprocessor.preprocess("This is a test sentence."))
    print(preprocessor.preprocess("This is another test sentence."))