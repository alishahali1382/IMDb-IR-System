import re
from typing import List


class SpellCorrection:
    def __init__(self, all_documents):
        """
        Initialize the SpellCorrection

        Parameters
        ----------
        all_documents : list of str
            The input documents.
        """
        self.all_shingled_words, self.word_counter = self.shingling_and_counting(all_documents)

    def shingle_word(self, word, k=2):
        """
        Convert a word into a set of shingles.

        Parameters
        ----------
        word : str
            The input word.
        k : int
            The size of each shingle.

        Returns
        -------
        set
            A set of shingles.
        """
        shingles = set()
        for i in range(len(word) - k + 1):
            shingles.add(word[i:i + k])
        return shingles

    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score.

        Parameters
        ----------
        first_set : set
            First set of shingles.
        second_set : set
            Second set of shingles.

        Returns
        -------
        float
            Jaccard score.
        """
        intersection = len(first_set.intersection(second_set))
        union = len(first_set) + len(second_set) - intersection
        return intersection / union

    def shingling_and_counting(self, all_documents: List[str]):
        """
        Shingle all words of the corpus and count TF of each word.

        Parameters
        ----------
        all_documents : list of str
            The input documents.

        Returns
        -------
        all_shingled_words : dict
            A dictionary from words to their shingle sets.
        word_counter : dict
            A dictionary from words to their TFs.
        """
        all_shingled_words = dict()
        word_counter = dict()
        for document in all_documents:
            words = re.findall(r'\w+', document.lower())
            for word in words:
                if word not in word_counter:
                    word_counter[word] = 0
                    all_shingled_words[word] = self.shingle_word(word)
                word_counter[word] += 1
        return all_shingled_words, word_counter

    def find_nearest_words(self, word: str):
        """
        Find correct form of a misspelled word.

        Parameters
        ----------
        word : str
            The misspelled word.

        Returns
        -------
        list of str
            5 nearest words.
        """
        print("Finding nearest words for:", word)
        word_shingles = self.shingle_word(word)
        candidates = []
        for candidate, candidate_shingles in self.all_shingled_words.items():
            jaccard = self.jaccard_score(word_shingles, candidate_shingles)
            candidates.append((candidate, jaccard))

        candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = candidates[:5]
        
        max_tf = max(self.word_counter[candidate] for candidate, _ in candidates)
        for i, (candidate, score) in enumerate(candidates):
            normalized_tf = self.word_counter[candidate] / max_tf
            candidates[i] = (candidate, score * normalized_tf)

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [candidate for candidate, _ in candidates]

    def spell_check(self, query: str):
        """
        Find correct form of a misspelled query.

        Parameters
        ----------
        query : str
            The misspelled query.

        Returns
        -------
        str
            Correct form of the query.
        """
        result = []
        words = re.findall(r'\w+', query.lower())
        for word in words:
            if word not in self.all_shingled_words:
                candidates = self.find_nearest_words(word)
                result.append(candidates[0])
            else:
                result.append(word)
        return ' '.join(result)