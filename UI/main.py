import streamlit as st
import sys

sys.path.append("../")
from Logic import utils
import time
from enum import Enum
import random
from Logic.core.utility.snippet import Snippet
from Logic.core.link_analysis.analyzer import LinkAnalyzer
from Logic.utils import main_movies_dataset as movies_dataset
from UI.poster_handler import get_movie_poster

# Initialize Snippet object
snippet_obj = Snippet()

with open("styles.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Enum for colors
class Color(Enum):
    RED = "#FF4C4C"
    GREEN = "#23D160"
    BLUE = "#3273DC"
    YELLOW = "#FFDD57"
    WHITE = "#FFFFFF"
    CYAN = "#0AN8B0"
    MAGENTA = "#B86BFF"

def get_top_x_movies_by_rank(x: int, results: list):
    corpus = []
    root_set = []
    document_index = {movie['id']: movie for movie in movies_dataset}
    for movie_id, movie_detail in document_index.items():
        movie_title = movie_detail["title"]
        stars = movie_detail["stars"]
        corpus.append({"id": movie_id, "title": movie_title, "stars": stars})

    for element in results:
        movie_id = element[0]
        movie_detail = document_index[movie_id]
        movie_title = movie_detail["title"]
        stars = movie_detail["stars"]
        root_set.append({"id": movie_id, "title": movie_title, "stars": stars})

    analyzer = LinkAnalyzer(root_set=root_set)
    analyzer.expand_graph(corpus=corpus)
    actors, movies = analyzer.hits(max_result=x)
    return actors, movies

# Function to highlight search terms in the summary
def get_summary_with_snippet(movie_info, query):
    summary = movie_info["first_page_summary"]
    snippet, not_exist_words = snippet_obj.find_snippet(summary, query)
    if "***" in snippet:
        snippet = snippet.split()
        for i in range(len(snippet)):
            current_word = snippet[i]
            if current_word.startswith("***") and current_word.endswith("***"):
                current_word_without_star = current_word[3:-3]
                summary = summary.lower().replace(
                    current_word_without_star,
                    f"<b><font color={random.choice(list(Color)).value}>{current_word_without_star}</font></b>",
                )
    return summary

def search_time(start, end):
    st.success("Search took: {:.6f} milliseconds".format((end - start) * 1e3))

def show_movie_card(card, info, search_term):
    with card[0].container():
        release_year = info['release_year']
        if release_year.count('-') == 2:
            release_year = release_year.split('-')[0]
        st.markdown(f"### [{info['title']} ({release_year})]({info['URL']})", unsafe_allow_html=True)
        st.write(f"IMDb Score: {info.get('rating', 'N/A')}")
        st.markdown(
            f"<b>Summary:</b> {get_summary_with_snippet(info, search_term)}",
            unsafe_allow_html=True,
        )

        st.markdown("**Directors:**")
        for director in info["directors"]:
            st.text(director)

        st.markdown("**Stars:**")
        stars = info["stars"]
        if len(stars) > 5:
            with st.expander("View all actors"):
                for star in stars:
                    st.text(star)
        else:
            for star in stars:
                st.text(star)

        st.markdown("**Genres:**")
        genres = info["genres"].split()
        genres_str = ""
        for genre in genres:
            genres_str += f"<span style='color:{Color.RED.value}'>{genre}</span>, "
        genres_str = genres_str[:-2]
        st.markdown(genres_str, unsafe_allow_html=True)

        st.markdown("**Reviews:**")
        reviews = info.get("reviews", [])
        if reviews:
            with st.expander("Click to show reviews"):
                for review in reviews:
                    content, score = review
                    if score is not None:
                        st.markdown(f"**Review Score:** {score}")
                    if isinstance(content, list):
                        content = " ".join(content)
                    st.markdown(content)
                    st.markdown("---")

    with card[1].container():
        st.image(get_movie_poster(info["id"]), use_column_width=True)

    st.divider()


def search_handling(
    search_button,
    search_term,
    search_max_num,
    search_weights,
    search_method,
    unigram_smoothing,
    alpha,
    lamda,
    filter_button,
    num_filter_results,
):
    if filter_button:
        if "search_results" in st.session_state:
            top_actors, top_movies = get_top_x_movies_by_rank(
                num_filter_results, st.session_state["search_results"]
            )
            st.markdown(f"**Top {num_filter_results} Actors:**")
            for actor in top_actors:
                st.markdown(f"<span style='color:{Color.BLUE.value}'>{actor}</span>", unsafe_allow_html=True)
            st.divider()

        st.markdown(f"**Top {num_filter_results} Movies:**")
        for i in range(len(top_movies)):
            card = st.columns([3, 1])
            info = utils.get_movie_by_id(top_movies[i])
            show_movie_card(card, info, search_term)
        return

    if search_button:
        corrected_query = utils.correct_text(search_term)

        if corrected_query != search_term:
            st.warning(f"Your search terms were corrected to: {corrected_query}")
            search_term = corrected_query

        with st.spinner("Searching..."):
            time.sleep(0.5)  # for showing the spinner! (can be removed)
            start_time = time.time()
            result = utils.search(
                search_term,
                search_max_num,
                search_method,
                search_weights,
                unigram_smoothing=unigram_smoothing,
                alpha=alpha,
                lamda=lamda,
            )

            if "search_results" in st.session_state:
                st.session_state["search_results"] = result

            end_time = time.time()
            if len(result) == 0:
                st.warning("No results found!")
                return

            search_time(start_time, end_time)

            for i in range(len(result)):
                card = st.columns([3, 1])
                info = utils.get_movie_by_id(result[i][0])
                show_movie_card(card, info, search_term)

        st.session_state["search_results"] = result
        if "filter_state" in st.session_state:
            st.session_state["filter_state"] = (
                "search_results" in st.session_state
                and len(st.session_state["search_results"]) > 0
            )

def main():
    st.title("🎬 IMDB Movie Search Engine")
    st.markdown(
        '<span style="color:#FFD700">Developed By: MIR Team at Sharif University</span>',
        unsafe_allow_html=True,
    )

    search_term = st.text_input("🔍 Search Term")
    with st.expander("Advanced Search"):
        search_max_num = st.number_input(
            "Maximum number of results", min_value=5, max_value=100, value=10, step=5
        )
        weight_stars = st.slider(
            "Weight of stars in search",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1,
        )

        weight_genres = st.slider(
            "Weight of genres in search",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1,
        )

        weight_summary = st.slider(
            "Weight of summary in search",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1,
        )
        slider_ = st.slider("Select the number of top movies to show", 1, 10, 5)

        search_weights = [weight_stars, weight_genres, weight_summary]
        search_method = st.selectbox(
            "Search method", ("ltn.lnn", "ltc.lnc", "OkapiBM25", "unigram")
        )

        unigram_smoothing = None
        alpha, lamda = None, None
        if search_method == "unigram":
            unigram_smoothing = st.selectbox(
                "Smoothing method",
                ("naive", "bayes", "mixture"),
            )
            if unigram_smoothing == "bayes":
                alpha = st.slider(
                    "Alpha",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                )
            if unigram_smoothing == "mixture":
                alpha = st.slider(
                    "Alpha",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                )
                lamda = st.slider(
                    "Lambda",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                )

    if "search_results" not in st.session_state:
        st.session_state["search_results"] = []

    st.markdown('<style>div.stButton > button:first-child {text-align: center; width: 100%;}</style>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])

    with col1:
        search_button = st.button("🔍 Search!")
    with col2:
        filter_button = st.button("📊 Filter movies by ranking")

    search_handling(
        search_button,
        search_term,
        search_max_num,
        search_weights,
        search_method,
        unigram_smoothing,
        alpha,
        lamda,
        filter_button,
        slider_,
    )


if __name__ == "__main__":
    main()
