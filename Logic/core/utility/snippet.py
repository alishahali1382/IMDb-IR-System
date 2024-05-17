import re
from typing import List, Tuple
from Logic.core.utility.preprocess import Preprocessor

preprocessor = Preprocessor()

class Snippet:
    def __init__(self, number_of_words_on_each_side=5):
        """
        Initialize the Snippet

        Parameters
        ----------
        number_of_words_on_each_side : int
            The number of words on each side of the query word in the doc to be presented in the snippet.
        """
        self.number_of_words_on_each_side = number_of_words_on_each_side

    def remove_stop_words_from_query(self, query: str) -> List[str]:
        """
        Remove stop words from the input string and tokenize.

        Parameters
        ----------
        query : str
            The query that you need to delete stop words from.

        Returns
        -------
        list[str]
            The tokenized query without stop words.
        """
        return preprocessor.preprocess(query)

    def find_snippet(self, doc: str, query: str) -> Tuple[str, List[str]]:
        """
        Find snippet in a doc based on a query.

        Parameters
        ----------
        doc : str
            The retrieved doc which the snippet should be extracted from that.
        query : str
            The query which the snippet should be extracted based on that.

        Returns
        -------
        final_snippet : str
            The final extracted snippet. IMPORTANT: The keyword should be wrapped by *** on both sides.
            For example: Sahwshank ***redemption*** is one of ... (for query: redemption)
        not_exist_words : list
            Words in the query which don't exist in the doc.
        """
        query_tokens = self.remove_stop_words_from_query(query)
        doc_tokens = doc.split()
        posting_lists = {token: [] for token in query_tokens}
        for i, token in enumerate(preprocessor.normalize(doc).split()):
            token = preprocessor.remove_punctuations(token)
            if token in posting_lists:
                posting_lists[token].append(i)

        not_exist_words = [token for token in query_tokens if len(posting_lists[token]) == 0]
        final_snippet = ""

        all_positions = []
        for token, positions in posting_lists.items():
            for position in positions:
                all_positions.append((position, token))

        all_positions.sort()
        best_segment = None
        counts = {token: 0 for token in query_tokens if token not in not_exist_words}
        left = 0
        for right in range(len(all_positions)):
            counts[all_positions[right][1]] += 1
            while min(counts.values()) > 0:
                if best_segment is None or all_positions[right][0] - all_positions[left][0] < best_segment[1] - best_segment[0]:
                    best_segment = (all_positions[left][0], all_positions[right][0])
                counts[all_positions[left][1]] -= 1
                left += 1

        mark = [False] * len(doc_tokens)
        for token in query_tokens:
            for position in posting_lists[token]:
                if best_segment[0] <= position <= best_segment[1]:
                    doc_tokens[position] = "***" + doc_tokens[position] + "***"
                    left_mark = max(0, position - self.number_of_words_on_each_side)
                    right_mark = min(len(doc_tokens), position + self.number_of_words_on_each_side + 1)
                    for i in range(left_mark, right_mark):
                        mark[i] = True
                    break  # NOTE: only mark the first occurrence

        for i in range(len(doc_tokens)):
            if mark[i]:
                final_snippet += doc_tokens[i] + " "
            elif not final_snippet.endswith("..."):
                final_snippet += "..."

        return final_snippet, not_exist_words


if __name__ == '__main__':
    doc = "The Shawshank Redemption is a 1994 American drama film written and directed by Frank Darabont, based on the 1982 Stephen King novella Rita Hayworth and Shawshank Redemption."
    query = "redemption test"

    snippet = Snippet()
    final_snippet, not_exist_words = snippet.find_snippet(doc, query)
    print(final_snippet)
    print(not_exist_words)
