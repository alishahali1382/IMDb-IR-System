import json
import os
from typing import Dict

from Logic.core.search import SearchEngine
from Logic.core.utility.spell_correction import SpellCorrection
from Logic.core.utility.snippet import Snippet
from Logic.core.indexer.indexes_enum import Indexes

def load_movies_dataset(concat_fuck = True):
    """
    Load the movies dataset

    Returns
    -------
    List[dict]
        The movies dataset
    """
    def concat_shit(shit):
        if shit is None:
            return "" if concat_fuck else []
        return " ".join(shit) if concat_fuck else shit

    def load_movie():
        path = "crawled_data/"
        if os.getcwd().endswith("UI"):
            path = f"../{path}"
        
        current_file = 1
        while True:
            try:
                with open(f"{path}IMDB_crawled_{current_file:02}.json", "r") as f:
                    data = json.load(f)
                    for movie in data:
                        movie["stars"] = movie["stars"]
                        movie["genres"] = concat_shit(movie["genres"])
                        movie["summaries"] = concat_shit(movie["summaries"])
                        yield movie
                current_file += 1
            except FileNotFoundError as err:
                if current_file == 1:
                    raise Exception("No movie was found")
                return
    
    return list(load_movie())

movies_dataset = load_movies_dataset()
main_movies_dataset = load_movies_dataset(False)
search_engine = None
spell_correction_obj = None

print("Number of loaded movies:",len(movies_dataset))

def correct_text(text: str) -> str:
    global spell_correction_obj, movies_dataset
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
    text = text.strip()
    if spell_correction_obj is None:
        all_documents = [movie["summaries"] for movie in movies_dataset]
        spell_correction_obj = SpellCorrection(all_documents)

    return spell_correction_obj.spell_check(text)


def search(
    query: str,
    max_result_count: int,
    method: str = "ltn-lnn",
    weights: list = [0.3, 0.3, 0.4],
    should_print=False,
    preferred_genre: str = None,
    unigram_smoothing=None,
    alpha=None,
    lamda=None,
):
    global search_engine
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

    method: 'ltn.lnn' or 'ltc.lnc' or 'OkapiBM25' or 'unigram'

    preferred_genre: A list containing preference rates for each genre. If None, the preference rates are equal.

    Returns
    ----------------------------------------------------------------------------------------------------
    list
    Retrieved documents with snippet
    """
    if search_engine is None:
        search_engine = SearchEngine()
    weights = {
        Indexes.STARS: weights[0],
        Indexes.GENRES: weights[1],
        Indexes.SUMMARIES: weights[2],
    }
    return search_engine.search(
        query, method, weights, max_results=max_result_count, safe_ranking=True, smoothing_method=unigram_smoothing, alpha=alpha, lamda=lamda
    )


def get_movie_by_id(id: str) -> Dict[str, str]:
    global movies_dataset
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
    }
    for movie in movies_dataset:
        if movie["id"] != id:
            continue
        movie["URL"] = f"https://www.imdb.com/title/{movie['id']}"
        return movie
    return default
