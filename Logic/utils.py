from typing import Dict, List
from Logic.core.search import SearchEngine
from Logic.core.spell_correction import SpellCorrection
from Logic.core.snippet import Snippet
from Logic.core.indexer.indexes_enum import Indexes, Index_types
import json

def load_movies_dataset():
    """
    Load the movies dataset

    Returns
    -------
    List[dict]
        The movies dataset
    """
    def load_movie():
        current_file = 1
        while True:
            try:
                with open(f"crawled_data/IMDB_crawled_{current_file:02}.json", "r") as f:
                    data = json.load(f)
                    for movie in data:
                        yield movie
                current_file += 1
            except FileNotFoundError as err:
                return
    
    return list(load_movie())

movies_dataset = load_movies_dataset()
all_documents = [" ".join(movie["summaries"]) for movie in movies_dataset]
search_engine = SearchEngine()
spell_correction_obj = SpellCorrection(all_documents)

print(len(movies_dataset))

def correct_text(text: str) -> str:
    """
    Correct the give query text, if it is misspelled using Jacard similarity

    Paramters
    ---------
    text: str
        The query text

    Returns
    str
        The corrected form of the given text
    """
    return spell_correction_obj.spell_check(text)


def search(
    query: str,
    max_result_count: int,
    method: str = "ltn-lnn",
    weights: list = [0.3, 0.3, 0.4],
    should_print=False,
    preferred_genre: str = None,
):
    """
    Finds relevant documents to query

    Parameters
    ---------------------------------------------------------------------------------------------------
    max_result_count: Return top 'max_result_count' docs which have the highest scores.
                      notice that if max_result_count = -1, then you have to return all docs

    mode: 'detailed' for searching in title and text separately.
          'overall' for all words, and weighted by where the word appears on.

    where: when mode ='detailed', when we want search query
            in title or text not both of them at the same time.

    method: 'ltn.lnn' or 'ltc.lnc' or 'OkapiBM25'

    preferred_genre: A list containing preference rates for each genre. If None, the preference rates are equal.

    Returns
    ----------------------------------------------------------------------------------------------------
    list
    Retrieved documents with snippet
    """
    weights = ...  # TODO
    return search_engine.search(
        query, method, weights, max_results=max_result_count, safe_ranking=True
    )


def get_movie_by_id(id: str, movies_dataset: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Get movie by its id

    Parameters
    ---------------------------------------------------------------------------------------------------
    id: str
        The id of the movie

    movies_dataset: List[Dict[str, str]]
        The dataset of movies

    Returns
    ----------------------------------------------------------------------------------------------------
    dict
        The movie with the given id
    """
    default = {
        "Title": "404",
        "Summary": "Movie is not found, so enjoy Shawshank",
        "URL": "https://www.imdb.com/title/tt0111161/",
        "Cast": ["Morgan Freeman", "Tim Robbins"],
        "Genres": ["Drama", "Crime"],
        "Image_URL": "https://m.media-amazon.com/images/M/MV5BNDE3ODcxYzMtY2YzZC00NmNlLWJiNDMtZDViZWM2MzIxZDYwXkEyXkFqcGdeQXVyNjAwNDUxODI@._V1_.jpg",
    }
    for movie in movies_dataset:
        if movie["id"] != id:
            continue
        movie["Image_URL"] = (
            "https://m.media-amazon.com/images/M/MV5BNDE3ODcxYzMtY2YzZC00NmNlLWJiNDMtZDViZWM2MzIxZDYwXkEyXkFqcGdeQXVyNjAwNDUxODI@._V1_.jpg"  # a default picture for selected movies
        )
        movie["URL"] = (
            f"https://www.imdb.com/title/{movie['id']}"  # The url pattern of IMDb movies
        )
        return movie
    return default
