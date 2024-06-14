import concurrent.futures
import imdb
import streamlit as st

MAX_WORKERS = 5
PLACEHOLDER = "https://via.placeholder.com/200"

@st.cache_resource
def _get_imdb_instance():
    return imdb.IMDb()

def _get_movie_poster(imdb_id):
    ia = _get_imdb_instance()
    movie = ia.get_movie(imdb_id[2:])
    return movie.get('cover url', None)

@st.cache_resource
def _get_executer():
    return concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS)

@st.cache_resource
def _fetch_poster_async(imdb_id):
    future = _get_executer().submit(_get_movie_poster, imdb_id)
    return future

def get_movie_poster(imdb_id):
    future = _fetch_poster_async(imdb_id)
    if not future.done():
        return PLACEHOLDER
    try:
        return future.result()
    except Exception as e:
        st.write(f"Exception for ID {imdb_id}: {e}")
        return PLACEHOLDER
