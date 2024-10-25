import json
from .indexes_enum import Indexes, Index_types

class DocumentLengthsIndex:
    def __init__(self, movies_dataset, path='index/'):
        """
        Initializes the DocumentLengthsIndex class.

        Parameters
        ----------
        path : str
            The path to the directory where the indexes are stored.

        """

        # self.documents_index = Index_reader(path, index_name=Indexes.DOCUMENTS).index
        self.documents_index = {movie['id']: movie for movie in movies_dataset}
        self.document_length_index = {
            Indexes.STARS: self.get_documents_length(Indexes.STARS.value),
            Indexes.GENRES: self.get_documents_length(Indexes.GENRES.value),
            Indexes.SUMMARIES: self.get_documents_length(Indexes.SUMMARIES.value)
        }
        self.store_document_lengths_index(path, Indexes.STARS)
        self.store_document_lengths_index(path, Indexes.GENRES)
        self.store_document_lengths_index(path, Indexes.SUMMARIES)

    def get_documents_length(self, where: str):
        """
        Gets the documents' length for the specified field.

        Parameters
        ----------
        where : str
            The field to get the document lengths for.

        Returns
        -------
        dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field (where).
        """

        ret = {}
        for doc_id, doc in self.documents_index.items():
            ret[doc_id] = len(doc[where])
        
        return ret
    
    def store_document_lengths_index(self, path , index_name):
        """
        Stores the document lengths index to a file.

        Parameters
        ----------
        path : str
            The path to the directory where the indexes are stored.
        index_name : Indexes
            The name of the index to store.
        """
        path = path + index_name.value + '_' + Index_types.DOCUMENT_LENGTH.value + '_index.json'
        with open(path, 'w') as file:
            json.dump(self.document_length_index[index_name], file, indent=4)
    

if __name__ == '__main__':
    from Logic.utils import movies_dataset
    document_lengths_index = DocumentLengthsIndex(movies_dataset)
    print('Document lengths index stored successfully.')