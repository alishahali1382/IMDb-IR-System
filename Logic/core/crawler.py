from contextlib import contextmanager
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, wait
from queue import Queue
from threading import Lock
from typing import List, Set

import requests
from bs4 import BeautifulSoup


@contextmanager
def SessionPool(max_sessions):
    session_queue = Queue(maxsize=max_sessions)
    for _ in range(max_sessions):
        session_queue.put(requests.Session())

    yield session_queue

    while not session_queue.empty():
        session = session_queue.get()
        session.close()


class IMDbCrawler:
    """
    put your own user agent in the headers
    """

    headers = {
        "authority": "www.imdb.com",
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "accept-language": "en-US,en;q=0.9",
        "cache-control": "max-age=0",
        "sec-ch-ua": '"Not(A:Brand";v="24", "Chromium";v="122"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Linux"',
        "sec-fetch-dest": "document",
        "sec-fetch-mode": "navigate",
        "sec-fetch-site": "same-origin",
        "sec-fetch-user": "?1",
        "upgrade-insecure-requests": "1",
        "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    }
    top_250_URL = "https://www.imdb.com/chart/top/"

    def __init__(self, crawling_threshold=1000):
        """
        Initialize the crawler

        Parameters
        ----------
        crawling_threshold: int
            The number of pages to crawl
        """
        # TODO
        self.crawling_threshold = crawling_threshold
        self.not_crawled: Queue[str] = None
        self.crawled: List[dict] = []
        self.crawled_lock = Lock()
        self.added_ids: Set[str] = set()
        self.added_ids_lock = Lock()
        self.session_pool: Queue[requests.Session] = None

    @staticmethod
    def get_id_from_URL(URL: str):
        """
        Get the id from the URL of the site. The id is what comes exactly after title.
        for example the id for the movie https://www.imdb.com/title/tt0111161/?ref_=chttp_t_1 is tt0111161.

        Parameters
        ----------
        URL: str
            The URL of the site
        Returns
        ----------
        str
            The id of the site
        """
        return re.match(r"/title/(.*?)/", URL).group(1)

    @staticmethod
    def get_movie_URL_from_id(id: str):
        """
        Get the URL of the movie from the id. The URL is what comes exactly after title.
        for example the URL for the movie tt0111161 is https://www.imdb.com/title/tt0111161/?ref_=chttp_t_1.

        Parameters
        ----------
        id: str
            The id of the site
        Returns
        ----------
        str
            The URL of the site
        """
        return f"https://www.imdb.com/title/{id}/?ref_=chttp_t_1"

    def write_to_file_as_json(self):
        """
        Save the crawled files into json
        """
        # TODO
        pass

    def read_from_file_as_json(self):
        """
        Read the crawled files from json
        """
        # TODO
        with open("IMDB_crawled.json", "r") as f:
            self.crawled = None

        with open("IMDB_not_crawled.json", "w") as f:
            self.not_crawled = None

        self.added_ids = None

    def get(self, URL: str) -> requests.Response:
        """
        Make a get request to the URL through session pool and return the response

        Parameters
        ----------
        URL: str
            The URL of the site
        Returns
        ----------
        requests.Response
            The response of the get request
        """
        try:
            session = self.session_pool.get()
            return session.get(URL, headers=self.headers)
        finally:
            self.session_pool.put(session)

    def crawl(self, URL: str) -> BeautifulSoup:
        """
        Make a get request to the URL and return the soup

        Parameters
        ----------
        URL: str
            The URL of the site
        Returns
        ----------
        bs4.BeautifulSoup
            The parsed content of the page
        """
        if URL == self.top_250_URL:
            with open("temp.html", "r") as f:
                return BeautifulSoup(f.read(), features="html.parser")

        if session is None:
            session = self.default_session
        resp = self.get(URL)
        if resp.status_code != 200:
            logging.error(f"Failed to get {URL}. status={resp.status_code}")
            return None
        soup = BeautifulSoup(resp.content, features="html.parser")
        return soup

    def extract_top_250(self):
        """
        Extract the top 250 movies from the top 250 page and use them as seed for the crawler to start crawling.
        """
        soup = self.crawl(self.top_250_URL)
        list_tag = soup.find("div", {"data-testid": "chart-layout-main-column"})
        top250 = list_tag.find_all("a", class_="ipc-title-link-wrapper")
        for tag in top250:
            href = tag.attrs["href"]
            id = self.get_id_from_URL(href)
            self.not_crawled.put(id)

    def get_imdb_instance(self):
        return {
            "id": None,  # str
            "title": None,  # str
            "first_page_summary": None,  # str
            "release_year": None,  # str
            "mpaa": None,  # str
            "budget": None,  # str
            "gross_worldwide": None,  # str
            "rating": None,  # str
            "directors": None,  # List[str]
            "writers": None,  # List[str]
            "stars": None,  # List[str]
            "related_links": None,  # List[str]
            "genres": None,  # List[str]
            "languages": None,  # List[str]
            "countries_of_origin": None,  # List[str]
            "summaries": None,  # List[str]
            "synopsis": None,  # List[str]
            "reviews": None,  # List[List[str]]
        }

    def start_crawling(self):
        """
        Start crawling the movies until the crawling threshold is reached.

        ThreadPoolExecutor is used to make the crawler faster by using multiple threads to crawl the pages.
        You are free to use it or not. If used, not to forget safe access to the shared resources.
        """
        with SessionPool(20) as self.session_pool:
            self.extract_top_250()
            futures = []
            crawled_counter = 0

            with ThreadPoolExecutor(max_workers=20) as executor:
                while crawled_counter < self.crawling_threshold:
                    if self.not_crawled.empty():
                        if len(futures) == 0:
                            break
                        wait(futures)
                        futures = []
                    else:
                        id = self.not_crawled.get()
                        futures.append(executor.submit(self.crawl_page_info, id))
                        crawled_counter += 1
                        
            wait(futures)

    def crawl_page_info(self, id):
        """
        Main Logic of the crawler. It crawls the page and extracts the information of the movie.
        Use related links of a movie to crawl more movies.

        Parameters
        ----------
        id: str
            The id of the movie
        """
        print("new iteration", URL)
        
        # TODO
        pass

    def extract_movie_info(self, soup, movie, URL):
        """
        Extract the information of the movie from the response and save it in the movie instance.

        Parameters
        ----------
        soup: bs4.BeautifulSoup
            The parsed response of the get request
        movie: dict
            The instance of the movie
        URL: str
            The URL of the site
        """
        # TODO
        movie["title"] = None
        movie["first_page_summary"] = None
        movie["release_year"] = None
        movie["mpaa"] = None
        movie["budget"] = None
        movie["gross_worldwide"] = None
        movie["directors"] = None
        movie["writers"] = None
        movie["stars"] = None
        movie["related_links"] = self.get_related_links(soup)
        movie["genres"] = None
        movie["languages"] = None
        movie["countries_of_origin"] = None
        movie["rating"] = None
        movie["summaries"] = None
        movie["synopsis"] = None
        movie["reviews"] = None

    def get_summary_link(url):
        """
        Get the link to the summary page of the movie
        Example:
        https://www.imdb.com/title/tt0111161/ is the page
        https://www.imdb.com/title/tt0111161/plotsummary is the summary page

        Parameters
        ----------
        url: str
            The URL of the site
        Returns
        ----------
        str
            The URL of the summary page
        """
        try:
            # TODO
            pass
        except:
            print("failed to get summary link")

    def get_review_link(url):
        """
        Get the link to the review page of the movie
        Example:
        https://www.imdb.com/title/tt0111161/ is the page
        https://www.imdb.com/title/tt0111161/reviews is the review page
        """
        try:
            # TODO
            pass
        except:
            print("failed to get review link")

    def get_title(soup):
        """
        Get the title of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The title of the movie

        """
        try:
            # TODO
            pass
        except:
            print("failed to get title")

    def get_first_page_summary(soup):
        """
        Get the first page summary of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The first page summary of the movie
        """
        try:
            # TODO
            pass
        except:
            print("failed to get first page summary")

    def get_director(soup):
        """
        Get the directors of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The directors of the movie
        """
        try:
            # TODO
            pass
        except:
            print("failed to get director")

    def get_stars(soup):
        """
        Get the stars of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The stars of the movie
        """
        try:
            # TODO
            pass
        except:
            print("failed to get stars")

    def get_writers(soup):
        """
        Get the writers of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The writers of the movie
        """
        try:
            # TODO
            pass
        except:
            print("failed to get writers")

    @staticmethod
    def get_related_links(soup):
        """
        Get the related links of the movie from the More like this section of the page from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The related links of the movie
        """
        try:
            # TODO
            pass
        except:
            print("failed to get related links")

    def get_summary(soup):
        """
        Get the summary of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The summary of the movie
        """
        try:
            # TODO
            pass
        except:
            print("failed to get summary")

    def get_synopsis(soup):
        """
        Get the synopsis of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The synopsis of the movie
        """
        try:
            # TODO
            pass
        except:
            print("failed to get synopsis")

    def get_reviews_with_scores(soup):
        """
        Get the reviews of the movie from the soup
        reviews structure: [[review,score]]

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[List[str]]
            The reviews of the movie
        """
        try:
            # TODO
            pass
        except:
            print("failed to get reviews")

    def get_genres(soup):
        """
        Get the genres of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The genres of the movie
        """
        try:
            # TODO
            pass
        except:
            print("Failed to get generes")

    def get_rating(soup):
        """
        Get the rating of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The rating of the movie
        """
        try:
            # TODO
            pass
        except:
            print("failed to get rating")

    def get_mpaa(soup):
        """
        Get the MPAA of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The MPAA of the movie
        """
        try:
            # TODO
            pass
        except:
            print("failed to get mpaa")

    def get_release_year(soup):
        """
        Get the release year of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The release year of the movie
        """
        try:
            # TODO
            pass
        except:
            print("failed to get release year")

    def get_languages(soup):
        """
        Get the languages of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The languages of the movie
        """
        try:
            # TODO
            pass
        except:
            print("failed to get languages")
            return None

    def get_countries_of_origin(soup):
        """
        Get the countries of origin of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The countries of origin of the movie
        """
        try:
            # TODO
            pass
        except:
            print("failed to get countries of origin")

    def get_budget(soup):
        """
        Get the budget of the movie from box office section of the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The budget of the movie
        """
        try:
            # TODO
            pass
        except:
            print("failed to get budget")

    def get_gross_worldwide(soup):
        """
        Get the gross worldwide of the movie from box office section of the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The gross worldwide of the movie
        """
        try:
            # TODO
            pass
        except:
            print("failed to get gross worldwide")


def main():
    imdb_crawler = IMDbCrawler(crawling_threshold=600)
    # imdb_crawler.read_from_file_as_json()
    imdb_crawler.start_crawling()
    print("done")
    # imdb_crawler.write_to_file_as_json()


if __name__ == "__main__":
    # main()
    imdb_crawler = IMDbCrawler(crawling_threshold=5)
    imdb_crawler.extract_top_250()
