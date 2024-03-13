from contextlib import contextmanager
import json
import logging
import re
from contextvars import ContextVar
from concurrent.futures import ThreadPoolExecutor, wait
from queue import Queue
from threading import Lock
import threading
from typing import List, Optional, Set

import requests
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="[{asctime}][{levelname}][{threadName:<23}] {message}",
    style="{",
    handlers=[
        logging.FileHandler("IMDB_crawler.log"),
        logging.StreamHandler()
    ]
)

# @contextmanager
# def SessionPool(max_sessions):
#     session_queue = Queue(maxsize=max_sessions)
#     for _ in range(max_sessions):
#         session_queue.put(requests.Session())

#     yield session_queue

#     while not session_queue.empty():
#         session = session_queue.get()
#         session.close()


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
        self.not_crawled: Queue[str] = Queue()
        self.crawled: List[dict] = []
        self.crawled_lock = Lock()
        self.added_ids: Set[str] = set()
        self.added_ids_lock = Lock()
        # self.session_pool: Queue[requests.Session] = None
        self.session: ContextVar[requests.Session] = ContextVar("session")
        self.current_movie: ContextVar[str] = ContextVar("current_movie")

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
        with open("IMDB_crawled.json", "w") as f:
            json.dump(self.crawled, f)

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
        session = self.session.get(None)
        if session is None:
            print(f"created a session for thread: {threading.current_thread().name}")
            session = requests.Session()
            self.session.set(session)
        return session.get(URL, headers=self.headers, timeout=3)
        # try:
            # session = self.session_pool.get()
            # return session.get(URL, headers=self.headers, timeout=10)
        # finally:
        #     self.session_pool.put(session)

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
        # if URL == self.top_250_URL:
        with open("temp.html", "r") as f:
            return BeautifulSoup(f.read(), features="html.parser")

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
        self.add_to_crawling_queue([self.get_id_from_URL(tag.attrs["href"]) for tag in top250])

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

    def start_crawling(self, max_workers=20):
        """
        Start crawling the movies until the crawling threshold is reached.

        ThreadPoolExecutor is used to make the crawler faster by using multiple threads to crawl the pages.
        You are free to use it or not. If used, not to forget safe access to the shared resources.
        """
        # with SessionPool(max_workers) as self.session_pool:
        self.extract_top_250()
        futures = []
        crawled_counter = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # executor.map(self.crawl_page_info, self.not_crawled.queue)
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

    def add_to_crawling_queue(self, ids):
        """
        Add the ids to the crawling queue and the added ids set.
        """
        with self.added_ids_lock:
            for id in ids:
                if id not in self.added_ids:
                    self.not_crawled.put(id)
                    self.added_ids.add(id)

    def add_to_crawled(self, movie):
        """
        Add the movie to the crawled list.
        """
        with self.crawled_lock:
            self.crawled.append(movie)

    def crawl_page_info(self, id):
        """
        Main Logic of the crawler. It crawls the page and extracts the information of the movie.
        Use related links of a movie to crawl more movies.

        Parameters
        ----------
        id: str
            The id of the movie
        """
        self.current_movie.set(id)
        # logging.info(f"Started crawling {id}")
        url = self.get_movie_URL_from_id(id)
        movie = self.get_imdb_instance()
        movie["id"] = id
        soup = self.crawl(url)
        if soup is None:
            logging.error(f"Failed to crawl {url}")
            return
        
        self.extract_movie_info(soup, movie, url)
        self.add_to_crawled(movie)
        self.add_to_crawling_queue(movie["related_links"])

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
        json_data = json.loads(soup.find("script", type="application/ld+json").text)
        movie["title"] = self.get_title(json_data)
        movie["first_page_summary"] = self.get_first_page_summary(json_data)
        movie["release_year"] = self.get_release_year(json_data)
        movie["mpaa"] = self.get_mpaa(json_data)
        movie["budget"] = self.get_budget(soup)
        movie["gross_worldwide"] = self.get_gross_worldwide(soup)
        movie["directors"] = self.get_director(json_data)
        movie["writers"] = self.get_writers(json_data)
        movie["stars"] = self.get_stars(soup)
        movie["related_links"] = self.get_related_links(soup)
        movie["genres"] = self.get_genres(json_data)
        movie["languages"] = self.get_languages(soup)
        movie["countries_of_origin"] = self.get_countries_of_origin(soup)
        movie["rating"] = self.get_rating(json_data)
        movie["summaries"] = None
        movie["synopsis"] = None
        movie["reviews"] = None
        logging.info(f"Finished crawling {movie['title']}")

    @staticmethod
    def get_summary_link(url: str) -> str:
        """
        Get the link to the summary page of the movie
        Example:
        https://www.imdb.com/title/tt0111161/ is the page
        https://www.imdb.com/title/tt0111161/plotsummary/ is the summary page

        Parameters
        ----------
        url: str
            The URL of the site
        Returns
        ----------
        str
            The URL of the summary page
        """
        return url + "plotsummary/"

    @staticmethod
    def get_review_link(url: str) -> str:
        """
        Get the link to the review page of the movie
        Example:
        https://www.imdb.com/title/tt0111161/ is the page
        https://www.imdb.com/title/tt0111161/reviews/ is the review page
        """
        return url + "reviews/"

    def get_title(self, json_data: dict) -> Optional[str]:
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
            return json_data["name"]
        except:
            logging.error(f"failed to get title of movie {self.current_movie.get()}")

    def get_first_page_summary(self, json_data: dict) -> Optional[str]:
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
            return json_data["description"]
        except:
            logging.error(f"failed to get first page summary of movie {self.current_movie.get()}")

    def get_director(self, json_data: dict) -> Optional[List[str]]:
        """
        Get the directors of the movie from the soup

        Parameters
        ----------
        json_data: dict
            The json data of the page
        Returns
        ----------
        List[str]
            The directors of the movie
        """
        try:
            return [person["name"] for person in json_data["director"]]
        except:
            logging.error(f"failed to get directors of movie {self.current_movie.get()}")

    def get_stars(self, soup: BeautifulSoup) -> Optional[List[str]]:
        """
        Get the stars of the movie from the soup

        Parameters
        ----------
        json_data: dict
            The json data of the page
        Returns
        ----------
        List[str]
            The stars of the movie
        """
        try:
            tags = soup.find_all("a", {"data-testid": "title-cast-item__actor"})
            return [tag.text for tag in tags]
        except:
            logging.error(f"failed to get stars of movie {self.current_movie.get()}")

    def get_writers(self, json_data: dict) -> Optional[List[str]]:
        """
        Get the writers of the movie from the soup

        Parameters
        ----------
        json_data: dict
            The json data of the page
        Returns
        ----------
        List[str]
            The writers of the movie
        """
        try:
            return [person["name"] for person in json_data["creator"] if person["@type"] == "Person"]
        except:
            logging.error(f"failed to get writers of movie {self.current_movie.get()}")        

    def get_related_links(self, soup: BeautifulSoup) -> Optional[List[str]]:
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
            tag_related = soup.find("section", {"data-testid": "MoreLikeThis"})
            poster_cards = tag_related.find_all("div", class_="ipc-poster-card", role="group")
            return [self.get_id_from_URL(card.find("a").attrs["href"]) for card in poster_cards]
        except:
            logging.error(f"failed to get related links of movie {self.current_movie.get()}")

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

    def get_genres(self, json_data: dict) -> Optional[List[str]]:
        """
        Get the genres of the movie from the soup

        Parameters
        ----------
        json_data: dict
            The json data of the page
        Returns
        ----------
        List[str]
            The genres of the movie
        """
        try:
            return json_data["genre"]
        except:
            logging.error(f"failed to get genres of movie {self.current_movie.get()}")

    def get_rating(self, json_data: dict) -> Optional[str]:
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
            return json_data["aggregateRating"]["ratingValue"]
        except:
            logging.error(f"failed to get rating of movie {self.current_movie.get()}")

    def get_mpaa(self, json_data: dict) -> Optional[str]:
        """
        Get the MPAA of the movie from the soup

        Parameters
        ----------
        json_data: dict
            The json data of the page
        Returns
        ----------
        str
            The MPAA of the movie
        """
        try:
            return json_data["contentRating"]
        except:
            logging.error(f"failed to get MPAA of movie {self.current_movie.get()}")

    def get_release_year(self, json_data: dict) -> Optional[str]:
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
            return json_data["datePublished"]
            # return tag.text
        except:
            logging.error(f"failed to get release year of movie {self.current_movie.get()}")

    def get_languages(self, soup: BeautifulSoup) -> Optional[List[str]]:
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
            list_tag = soup.find("li", {"data-testid": "title-details-languages", "role": "presentation"})
            tags = list_tag.find_all("a", class_="ipc-metadata-list-item__list-content-item--link")
            return [tag.text for tag in tags]
        except:
            logging.error(f"failed to get languages of movie {self.current_movie.get()}")

    @staticmethod
    def remove_new_lines(text: str) -> str:
        return ' '.join(word for word in text.replace('\n', ' ').split(' ') if word)

    def get_countries_of_origin(self, soup: BeautifulSoup) -> Optional[List[str]]:
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
            list_tag = soup.find("li", {"data-testid": "title-details-origin", "role": "presentation"})
            tags = list_tag.find_all("a", class_="ipc-metadata-list-item__list-content-item--link")
            return [self.remove_new_lines(tag.text) for tag in tags]
        except:
            logging.error(f"failed to get countries of origin of movie {self.current_movie.get()}")

    def get_budget(self, soup: BeautifulSoup) -> Optional[str]:
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
            tag = soup.find("li", {"data-testid": "title-boxoffice-budget", "role": "presentation"})
            tag = tag.find("div", class_="ipc-metadata-list-item__content-container")            
            return tag.text.strip()
        except:
            logging.error(f"failed to get budget of movie {self.current_movie.get()}")

    def get_gross_worldwide(self, soup: BeautifulSoup) -> Optional[str]:
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
            tag = soup.find("li", {"data-testid": "title-boxoffice-cumulativeworldwidegross", "role": "presentation"})
            tag = tag.find("div", class_="ipc-metadata-list-item__content-container")
            tag = tag.find("span", class_="ipc-metadata-list-item__list-content-item")
            return tag.text.strip()
        except:
            logging.error(f"failed to get gross worldwide of movie {self.current_movie.get()}")


def main():
    imdb_crawler = IMDbCrawler(crawling_threshold=10)
    # imdb_crawler.read_from_file_as_json()
    imdb_crawler.start_crawling(max_workers=20)
    print("done")
    # imdb_crawler.write_to_file_as_json()

if __name__ == "__main__":
    # main()
    imdb_crawler = IMDbCrawler(crawling_threshold=50)
    imdb_crawler.crawl_page_info("tt0120737")
    from pprint import pprint
    pprint(imdb_crawler.crawled)
    # print(imdb_crawler.crawled[0]["title"])
    # imdb_crawler.crawl_page_info("tt0050083")
    # imdb_crawler.start_crawling()
