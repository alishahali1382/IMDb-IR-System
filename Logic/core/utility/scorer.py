from typing import List, Literal
import numpy as np


class Scorer:
    def __init__(self, index, number_of_documents):
        """
        Initializes the Scorer.

        Parameters
        ----------
        index : dict
            The index to score the documents with.
        number_of_documents : int
            The number of documents in the index.
        """

        self.index = index
        self.idf = {}
        self.N = number_of_documents

    def get_list_of_documents(self, query):
        """
        Returns a list of documents that contain at least one of the terms in the query.

        Parameters
        ----------
        query: List[str]
            The query to be scored

        Returns
        -------
        list
            A list of documents that contain at least one of the terms in the query.

        Note
        ---------
            The current approach is not optimal but we use it due to the indexing structure of the dict we're using.
            If we had pairs of (document_id, tf) sorted by document_id, we could improve this.
                We could initialize a list of pointers, each pointing to the first element of each list.
                Then, we could iterate through the lists in parallel.

        """
        list_of_documents = []
        for term in query:
            if term in self.index.keys():
                list_of_documents.extend(self.index[term].keys())
        return list(set(list_of_documents))
    
    def get_idf(self, term: str):
        """
        Returns the inverse document frequency of a term.

        Parameters
        ----------
        term : str
            The term to get the inverse document frequency for.

        Returns
        -------
        float
            The inverse document frequency of the term.

        Note
        -------
            It was better to store dfs in a separate dict in preprocessing.
        """
        idf = self.idf.get(term, None)
        if idf is None:
            self.idf[term] = idf = np.log10(self.N / len(self.index[term]))
        return idf

    def get_query_tfs(self, query: List[str]):
        """
        Returns the term frequencies of the terms in the query.

        Parameters
        ----------
        query : List[str]
            The query to get the term frequencies for.

        Returns
        -------
        dict
            A dictionary of the term frequencies of the terms in the query.
        """
        tfs= {term: 0 for term in query}
        for term in query:
            tfs[term] += 1
        return tfs

    def compute_scores_with_vector_space_model(self, query: List[str], method: str):
        """
        compute scores with vector space model

        Parameters
        ----------
        query: List[str]
            The query to be scored
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c))
            The method to use for searching.

        Returns
        -------
        dict
            A dictionary of the document IDs and their scores.
        """
        doc_method, query_method= method.split('.')
        query_tfs = self.get_query_tfs(query)
        return {
            doc_id: self.get_vector_space_model_score(query, query_tfs, doc_id, doc_method, query_method)
            for doc_id in self.get_list_of_documents(query)
        }

    def get_vector_space_model_score(
        self, query, query_tfs, document_id, document_method, query_method
    ):
        """
        Returns the Vector Space Model score of a document for a query.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        query_tfs : dict
            The term frequencies of the terms in the query.
        document_id : str
            The document to calculate the score for.
        document_method : str (n|l)(n|t)(n|c)
            The method to use for the document.
        query_method : str (n|l)(n|t)(n|c)
            The method to use for the query.

        Returns
        -------
        float
            The Vector Space Model score of the document for the query.
        """
        m = len(query)

        doc_vectors = np.zeros(m)
        query_vectors = np.zeros(m)
        
        for i, term in enumerate(query):
            d_tf = self.index.get(term, {}).get(document_id, 0)
            if d_tf == 0:
                continue

            if document_method[0] == 'l' and d_tf != 0:
                d_tf = np.log10(d_tf) + 1

            d_idf= 1
            if document_method[1] == 't':
                d_idf = self.get_idf(term)
            doc_vectors[i] = d_tf * d_idf

            q_tf= query_tfs[term]
            if query_method[0] == 'l':
                q_tf= np.log10(q_tf) + 1

            q_idf= 1
            if query_method[1] == 't':
                q_idf= self.get_idf(term)

            query_vectors[i] = q_tf * q_idf

        if document_method[2]=='c':
            doc_vectors /= np.linalg.norm(doc_vectors)

        if query_method[2]=='c':
            query_vectors /= np.linalg.norm(query_vectors)

        return np.dot(query_vectors, doc_vectors)


    def compute_socres_with_okapi_bm25(
        self, query, average_document_field_length, document_lengths
    ):
        """
        compute scores with okapi bm25

        Parameters
        ----------
        query: List[str]
            The query to be scored
        average_document_field_length : float
            The average length of the documents in the index.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.

        Returns
        -------
        dict
            A dictionary of the document IDs and their scores.
        """
        return {
            doc_id: self.get_okapi_bm25_score(query, doc_id, average_document_field_length, document_lengths)
            for doc_id in self.get_list_of_documents(query)
        }

    def get_okapi_bm25_score(
        self, query, document_id, average_document_field_length, document_lengths
    ):
        """
        Returns the Okapi BM25 score of a document for a query.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        document_id : str
            The document to calculate the score for.
        average_document_field_length : float
            The average length of the documents in the index.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.

        Returns
        -------
        float
            The Okapi BM25 score of the document for the query.
        """
        k1 = 1.5
        b = 0.75
        score = 0
        for term in query:
            f = self.index[term].get(document_id, 0)
            df = len(self.index[term])
            doc_len = document_lengths[document_id]
            idf = self.get_idf(term)
            score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * doc_len / average_document_field_length))
        return score

    def compute_score_with_unigram_model(
        self, query: str, smoothing_method: Literal['bayes', 'naive', 'mixture'], document_lengths=None, alpha=0.5, lamda=0.5
    ):
        """
        Calculates the scores for each document based on the unigram model.

        Parameters
        ----------
        query : str
            The query to search for.
        smoothing_method : str (bayes | naive | mixture)
            The method used for smoothing the probabilities in the unigram model.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.
        alpha : float, optional
            The parameter used in bayesian smoothing method. Defaults to 0.5.
        lamda : float, optional
            The parameter used in some smoothing methods to balance between the document
            probability and the collection probability. Defaults to 0.5.

        Returns
        -------
        float
            A dictionary of the document IDs and their scores.
        """
        return {
            doc_id: self.calculate_scores_with_unigram_model(
                query, doc_id, smoothing_method, document_lengths, alpha, lamda
            )
            for doc_id in self.get_list_of_documents(query)
        }

    def calculate_scores_with_unigram_model(
            self, query, document_id, smoothing_method, document_lengths, alpha, lamda
    ):
        """
        Calculates the scores for each document based on the unigram model.

        Parameters
        ----------
        query : str
            The query to search for.
        document_id : str
            The document to calculate the score for.
        smoothing_method : str (bayes | naive | mixture)
            The method used for smoothing the probabilities in the unigram model.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.
        alpha : float, optional
            The parameter used in bayesian smoothing method. Defaults to 0.5.
        lamda : float, optional
            The parameter used in some smoothing methods to balance between the document
            probability and the collection probability. Defaults to 0.5.

        Returns
        -------
        float
            The Unigram score of the document for the query.
        """
        methods = {
            'bayes': self.compute_unigram_bayes_score,
            'naive': self.compute_unigram_naive_score,
            'mixture': self.compute_unigram_mixture_score
        }
        query_tfs = self.get_query_tfs(query)
        return methods[smoothing_method](query, query_tfs, document_id, document_lengths, alpha, lamda)

    def compute_unigram_bayes_score(self, query, query_tfs, document_id, document_lengths, alpha, lamda):
        T = sum(document_lengths.values())
        doc_length = document_lengths[document_id]
        score = 0
        for term in query:
            cf = sum(self.index.get(term, {}).values())
            tf = self.index.get(term, {}).get(document_id, 0)
            score -= query_tfs[term] * np.log((tf + alpha * cf / T) / (doc_length + alpha))

        return score

    def compute_unigram_naive_score(self, query, query_tfs, document_id, document_lengths, alpha, lamda):
        T = len(self.index)
        doc_length = document_lengths[document_id]
        score = 0
        for term in query:
            tf = self.index.get(term, {}).get(document_id, 0)
            score -= query_tfs[term] * np.log((tf + 1/T) / (doc_length + 1))
        return score

    def compute_unigram_mixture_score(self, query, query_tfs, document_id, document_lengths, alpha, lamda):
        T = sum(document_lengths.values())
        doc_length = document_lengths[document_id]
        score = 0
        for term in query:
            tf = self.index.get(term, {}).get(document_id, 0)
            cf = sum(self.index.get(term, {}).values())
            score -= query_tfs[term] * np.log(lamda * tf / doc_length + (1 - lamda)*(cf / T))

        return score
