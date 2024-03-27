import itertools
import random
from typing import Iterator

import numpy as np


class MinHashLSH:
    def __init__(self, documents, num_hashes):
        """
        Initialize the MinHashLSH

        Parameters
        ----------
        documents : list of str
            The input documents for similarity analysis.
        num_hashes : int
            Number of hashes for mini-hashing.
        """
        self.documents = documents
        self.num_hashes = num_hashes
        self.num_docs = len(documents)
        self.hash_funcs = [self._make_hash_func(i) for i in range(num_hashes)]

    @staticmethod
    def _make_hash_func(i):
        return lambda x: hash(f"hash{i:03}<{x}>")

    def shingle_document(self, document: str, k=2):
        """
        Convert a document into a set of shingles.

        Parameters
        ----------
        document : str
            The input document.
        k : int
            The size of each shingle.

        Returns
        ----------
        set
            A set of shingles.
        """
        # return set(document[i : i + k] for i in range(len(document) - k + 1))  # character shingling
        words = document.split()
        return set(" ".join(words[i : i + k]) for i in range(len(words) - k + 1))  # word shingling

    def build_characteristic_matrix(self):
        """
        Build the characteristic matrix representing the presence of shingles in documents.

        Returns
        ----------
        numpy.ndarray
            The binary characteristic matrix.
        """
        # NOTE: with the corpus size, calling this function is stupid. I dont know why this is in template.
        shingle_sets = [self.shingle_document(doc) for doc in self.documents]
        universe = set.union(*shingle_sets)
        return np.array(
            [
                [1 if shingle in doc_shingles else 0 for doc_shingles in shingle_sets]
                for shingle in universe
            ],
            dtype=bool,
        )

    def min_hash_signature(self):
        """
        Perform Min-Hashing to generate hash signatures for documents.

        Returns
        ----------
        numpy.ndarray
            The Min-Hash signatures matrix.
        """
        for doc in self.documents:
            yield np.array(
                [
                    min(map(hash_func, self.shingle_document(doc)))
                    for hash_func in self.hash_funcs
                ]
            )

    def lsh_buckets(self, signatures: Iterator[np.ndarray], bands=10, rows_per_band=10):
        """
        Group documents into Locality-Sensitive Hashing (LSH) buckets based on Min-Hash signatures.

        Parameters
        ----------
        signatures : Iterator[np.ndarray]
            Min-Hash signatures for documents.
        bands : int
            Number of bands for LSH.
        rows_per_band : int
            Number of rows per band.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        buckets = {}
        for i, sig in enumerate(signatures):
            for band in range(bands):
                band_hash = hash(tuple(sig[band * rows_per_band : (band + 1) * rows_per_band]))
                if band_hash not in buckets:
                    buckets[band_hash] = []
                buckets[band_hash].append(i)
            
        return buckets

    def perform_lsh(self):
        """
        Perform the entire Locality-Sensitive Hashing (LSH) process.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        bands = self.num_hashes // 5
        return self.lsh_buckets(self.min_hash_signature(), bands, 5)

    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score for two sets.

        Parameters
        ----------
        first_set : set
            Set of first shingled document.
        second_set : set
            Set of second shingled document.

        Returns
        ----------
        float
            Jaccard score.
        """
        intersection_size = len(first_set.intersection(second_set))
        union_size = len(first_set.union(second_set))
        return intersection_size / union_size

    def jaccard_similarity_test(self, buckets, all_documents):
        """
        Test your near duplicate detection code based on jaccard similarity.

        Parameters
        ----------
        buckets : dict
            A dictionary mapping bucket IDs to lists of document indices.
        all_documents : list
            The input documents for similarity analysis.
        """
        correct_near_duplicates = 0
        all_near_duplicates = 0
        all_candidate_pairs = set()
        
        for bucket_id in buckets.keys():
            docs_in_this_bucket = buckets[bucket_id]
            unique_doc_ids = set(docs_in_this_bucket)
            all_candidate_pairs.update(itertools.combinations(unique_doc_ids, 2))

        for first_doc_id, second_doc_id in all_candidate_pairs:
            all_near_duplicates += 1

            first_shingled_doc = self.shingle_document(all_documents[first_doc_id], 2)
            second_shingled_doc = self.shingle_document(all_documents[second_doc_id], 2)

            near_duplicated_jaccard_score = self.jaccard_score(first_shingled_doc, second_shingled_doc)
            current_score = 0

            for _ in range(5):
                random_doc_id = first_doc_id
                while random_doc_id == first_doc_id or random_doc_id == second_doc_id:
                    random_doc_id = random.randint(0, len(all_documents) - 1)
                random_shingled_doc = self.shingle_document(all_documents[random_doc_id], 2)

                random_jaccard_score = self.jaccard_score(first_shingled_doc, random_shingled_doc)

                if near_duplicated_jaccard_score > random_jaccard_score:
                    current_score += 1

            if current_score == 5:
                correct_near_duplicates += 1

        # a good score is around 0.8
        print("your final score in near duplicate detection:", correct_near_duplicates / all_near_duplicates)

if __name__ == "__main__":
    import json
    with open("Logic/core/LSHFakeData.json", "r") as f:
        documents = json.load(f)
    document_summaries = [" # ".join(doc["summaries"]) for doc in documents]
    minhash_lsh = MinHashLSH(document_summaries, 500)
    minhash_lsh.jaccard_similarity_test(minhash_lsh.perform_lsh(), document_summaries)
