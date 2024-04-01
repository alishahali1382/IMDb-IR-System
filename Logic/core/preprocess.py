import re
from typing import List
import nltk

def _download_nltk():
    nltk.download('wordnet')
    nltk.download('punkt')

class Preprocessor:

    def __init__(self, documents: List[str]):
        """
        Initialize the class.

        Parameters
        ----------
        documents : list
            The list of documents to be preprocessed, path to stop words, or other parameters.
        """
        self.documents = documents
        stopwords_path = "/".join(__file__.split('/')[:-1]+['stopwords.txt'])
        with open(stopwords_path, 'r') as f:
            self.stopwords = set(f.read().split('\n'))

    def preprocess(self) -> List[str]:
        """
        Preprocess the text using the methods in the class.

        Returns
        ----------
        List[str]
            The preprocessed documents.
        """
        docs: List[str] = []
        for doc in self.documents:
            doc = self.normalize(doc)
            doc = self.remove_links(doc)
            doc = self.remove_punctuations(doc)
            words = self.tokenize(doc)
            words = self.remove_stopwords(words)
            docs.append(" ".join(words))
        return docs

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
        text = text.lower()
        text = nltk.stem.PorterStemmer().stem(text)
        text = nltk.stem.WordNetLemmatizer().lemmatize(text)
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
    preprocessor = Preprocessor(["This is a test sentence.", "This is another test sentence."])
    print(preprocessor.preprocess())