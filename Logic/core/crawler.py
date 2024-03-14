import json
import html
import logging
import re
from contextvars import ContextVar
from concurrent.futures import ThreadPoolExecutor, wait
from queue import Queue
from threading import Lock
import gc
from typing import List, Optional, Set
from logging.config import dictConfig

import requests
from bs4 import BeautifulSoup

dictConfig({
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {
        "detailed": {
            "format": "[{asctime}][{levelname}][{threadName:<23}] {message}",
            "style": "{"
        }
    },
    "handlers": {
        "file": {
            "class": logging.FileHandler,
            "filename": "crawler.log",
            "formatter": "detailed"
        },
        "file_error": {
            "class": logging.FileHandler,
            "filename": "crawler_error.log",
            "formatter": "detailed",
            "level": "ERROR"
        },
        "console": {
            "class": logging.StreamHandler,
            "formatter": "detailed"
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["file", "console", "file_error"]
    }   
})


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
        self.crawling_threshold = crawling_threshold
        self.not_crawled: Queue[str] = Queue()
        self.crawled: List[dict] = []
        self.crawled_lock = Lock()
        self.added_ids: Set[str] = set()
        self.added_ids_lock = Lock()
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

    def write_to_file_as_json(self, filename: str):
        """
        Save the crawled files into json
        """
        with open(filename, "w") as f:
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
            logging.info("created a session")
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

    def start_crawling(self, max_workers=10, file_batch_size=1000, write_to_file=True):
        """
        Start crawling the movies until the crawling threshold is reached.

        ThreadPoolExecutor is used to make the crawler faster by using multiple threads to crawl the pages.
        You are free to use it or not. If used, not to forget safe access to the shared resources.
        """
        # with SessionPool(max_workers) as self.session_pool:
        self.extract_top_250()
        futures = []
        crawled_counter = 0
        files_counter = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # executor.map(self.crawl_page_info, self.not_crawled.queue)
            while crawled_counter < self.crawling_threshold:
                if write_to_file and len(self.crawled) % file_batch_size == 0:
                    files_counter += 1
                    self.write_to_file_as_json(f"IMDB_crawled_{files_counter:02d}.json")
                    
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
        soup = self.crawl(url)
        if soup is None:
            logging.error(f"Failed to crawl {url}")
            return
        
        self.extract_movie_info(soup, movie, id)
        self.add_to_crawled(movie)
        # self.add_to_crawling_queue(movie["related_links"])

    def extract_movie_info(self, soup, movie, id):
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
        movie["id"] = id
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

        summary_soup = self.crawl(self.get_summary_link(id))
        plotsummary_json_data = json.loads(summary_soup.find("script", {"id": "__NEXT_DATA__", "type": "application/json"}).text)
        movie["summaries"] = self.get_summary(plotsummary_json_data)
        movie["synopsis"] = self.get_synopsis(plotsummary_json_data)

        reviews_soup = self.crawl(self.get_review_link(id))
        movie["reviews"] = self.get_reviews_with_scores(reviews_soup)
        logging.info(f"Finished crawling {movie['title']}")

    @staticmethod
    def get_summary_link(id: str) -> str:
        """
        Get the link to the summary page of the movie
        Example:
        https://www.imdb.com/title/tt0111161/ is the page
        https://www.imdb.com/title/tt0111161/plotsummary/ is the summary page

        Parameters
        ----------
        id: str
            The id of the site
        Returns
        ----------
        str
            The URL of the summary page
        """
        return f"https://www.imdb.com/title/{id}/plotsummary/"

    @staticmethod
    def get_review_link(id: str) -> str:
        """
        Get the link to the review page of the movie
        Example:
        https://www.imdb.com/title/tt0111161/ is the page
        https://www.imdb.com/title/tt0111161/reviews/ is the review page
        """
        return f"https://www.imdb.com/title/{id}/reviews/"

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

    @staticmethod
    def remove_html_suffix_from_text(text: str) -> str:
        # return re.sub(r'\u003cspan.+', '', text)
        # re.sub(r'<[^>]+>', '', dirty_text)
        while True:
            nex = html.unescape(text)
            nex = re.sub(r'<[^>]+>', '', nex)
            if nex == text:
                break
            text = nex
        return text

    def get_summary(self, json_data: dict) -> Optional[List[str]]:
        """
        Get the summary of the movie from the parsed json

        Parameters
        ----------
        json_data: dict
            The json data of the page
        Returns
        ----------
        List[str]
            The summary of the movie
        """
        try:
            for data in json_data['props']['pageProps']['contentData']['categories']:
                if data['id'] == 'summaries':
                    return [self.remove_html_suffix_from_text(item['htmlContent']) for item in data['section']['items']]
        except:
            logging.error(f"failed to get summary of movie {self.current_movie.get()}")

    def get_synopsis(self, json_data: dict) -> Optional[List[str]]:
        """
        Get the synopsis of the movie from the parsed json

        Parameters
        ----------
        json_data: dict
            The json data of the page
        Returns
        ----------
        List[str]
            The synopsis of the movie
        """
        try:
            for data in json_data['props']['pageProps']['contentData']['categories']:
                if data['id'] == 'synopsis':
                    return [self.remove_html_suffix_from_text(item['htmlContent']) for item in data['section']['items']]
        except:
            logging.error(f"failed to get synopsis of movie {self.current_movie.get()}")

    def get_reviews_with_scores(self, soup: BeautifulSoup) -> Optional[List[List[str]]]:
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
            tags = soup.find("div", class_="lister-list").find_all("div", class_="review-container")
            out = []
            for tag in tags:
                review = tag.find("div", class_="text show-more__control").text,
                score_tag = tag.find("span", class_="rating-other-user-rating")
                if score_tag is None:
                    # TODO: ask for intended behavior in this case
                    continue
                score = score_tag.find("span").text
                out.append([review, score])
            return out
        except Exception as e:
            logging.error(f"failed to get reviews of movie {self.current_movie.get()}: {e}")

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
        json_data: dict
            The json data of the page
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
    imdb_crawler = IMDbCrawler(crawling_threshold=10000)
    # imdb_crawler.read_from_file_as_json()
    imdb_crawler.start_crawling(max_workers=20, file_batch_size=1000, write_to_file=True)

if __name__ == "__main__":
    main()
    # imdb_crawler = IMDbCrawler(crawling_threshold=10)
    # imdb_crawler.crawl_page_info("tt0110912")
    # # from pprint import pprint
    # # pprint(imdb_crawler.crawled[0]["summaries"])
    # # print(imdb_crawler.crawled[0]["title"])
    # # imdb_crawler.crawl_page_info("tt0050083")
    # # imdb_crawler.start_crawling()
    # imdb_crawler.write_to_file_as_json()
