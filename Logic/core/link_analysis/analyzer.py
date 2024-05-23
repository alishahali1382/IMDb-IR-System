from .graph import LinkGraph

class LinkAnalyzer:
    def __init__(self, root_set):
        """
        Initialize the Link Analyzer attributes:

        Parameters
        ----------
        root_set: list
            A list of movie dictionaries with the following keys:
            "id": A unique ID for the movie
            "title": string of movie title
            "stars": A list of movie star names
        """
        self.root_set = root_set
        self.graph = LinkGraph()
        self.hubs = set()
        self.authorities = set()
        self.initiate_params()

    def initiate_params(self):
        """
        Initialize links graph, hubs list and authorities list based of root set

        Parameters
        ----------
        This function has no parameters. You can use self to get or change attributes
        """
        self.hubs = set(movie["id"] for movie in self.root_set)
        self.authorities = set(star for movie in self.root_set for star in movie["stars"])
        for movie in self.root_set:
            self.graph.add_node(movie["id"])
            for star in movie["stars"]:
                self.graph.add_node(star)
                self.graph.add_edge(movie["id"], star)

    def expand_graph(self, corpus):
        """
        expand hubs, authorities and graph using given corpus

        Parameters
        ----------
        corpus: list
            A list of movie dictionaries with the following keys:
            "id": A unique ID for the movie
            "stars": A list of movie star names

        Note
        ---------
        To build the base set, we need to add the hubs and authorities that are inside the corpus
        and refer to the nodes in the root set to the graph and to the list of hubs and authorities.
        """
        for movie in corpus:
            if movie["id"] not in self.hubs and self.authorities.intersection(set(movie["stars"])) != set():
                self.graph.add_node(movie["id"])
                self.hubs.add(movie["id"])

    def hits(self, num_iteration=10, max_result=10):
        """
        Return the top movies and actors using the Hits algorithm

        Parameters
        ----------
        num_iteration: int
            Number of algorithm execution iterations
        max_result: int
            The maximum number of results to return. If None, all results are returned.

        Returns
        -------
        list
            List of names of 10 actors with the most scores obtained by Hits algorithm in descending order
        list
            List of names of 10 movies with the most scores obtained by Hits algorithm in descending order
        """
        a_s = {actor: 1 for actor in self.authorities}
        h_s = {hub: 1 for hub in self.hubs}
        
        for i in range(num_iteration):

            for actor in a_s.keys():
                a_s[actor] = sum(h_s[movie] for movie in self.graph.get_predecessors(actor))

            a_s_sum = sum(a_s.values())
            a_s = {actor: score/a_s_sum for actor, score in a_s.items()}
            
            for movie in h_s.keys():
                h_s[movie] = sum(a_s[actor] for actor in self.graph.get_successors(movie))

            h_s_sum = sum(h_s.values())
            h_s = {movie: score/h_s_sum for movie, score in h_s.items()}

        actors = sorted(a_s, key=a_s.get, reverse=True)[:max_result]
        movies = sorted([movie['id'] for movie in self.root_set], key=h_s.get, reverse=True)[:max_result]
        return actors, movies

if __name__ == "__main__":
    # You can use this section to run and test the results of your link analyzer
    from Logic.utils import main_movies_dataset
    corpus = main_movies_dataset
    root_set = [
        movie for movie in main_movies_dataset
        if movie['rating'] and float(movie['rating']) >= 8
        and movie['languages'] and 'English' in movie['languages']
    ]
    print(f"length of root_set: {len(root_set)}")
    
    analyzer = LinkAnalyzer(root_set=root_set)
    analyzer.expand_graph(corpus=corpus)
    actors, movies = analyzer.hits(max_result=5)
    
    def get_movie_title(movie_id):
        for movie in root_set:
            if movie["id"] == movie_id:
                return movie["title"]
        return None
    
    print("Top Actors:")
    print(*actors, sep=' *** ')
    print("Top Movies:")
    print(*movies, sep=' *** ')
    print(*(get_movie_title(movie) for movie in movies), sep=' *** ')
    
