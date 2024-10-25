import matplotlib.pyplot as plt
import numpy as np
import random
import operator
import wandb

from typing import List, Tuple
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from collections import Counter
from .clustering_metrics import *


class ClusteringUtils:

    def cluster_kmeans(self, emb_vecs: List, n_clusters: int, max_iter: int = 100) -> Tuple[List, List]:
        """
        Clusters input vectors using the K-means method.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.
        n_clusters: int
            The number of clusters to form.

        Returns
        --------
        Tuple[List, List]
            Two lists:
            1. A list containing the cluster centers.
            2. A list containing the cluster index for each input vector.
        """
        emb_vecs = np.array(emb_vecs)

        cluster_centers = np.random.Generator(np.random.PCG64()).choice(emb_vecs, n_clusters, axis=0, replace=False)
        cluster_indices = np.zeros_like(emb_vecs, dtype=int)

        for i in range(max_iter):
            cluster_indices = np.argmin(np.linalg.norm(emb_vecs[:, None] - cluster_centers, axis=2), axis=1)
            cluster_centers = np.array([np.mean(emb_vecs[cluster_indices == k], axis=0) for k in range(n_clusters)])

        return cluster_centers, cluster_indices

    def get_most_frequent_words(self, documents: List[str], top_n: int = 10) -> List[Tuple[str, int]]:
        """
        Finds the most frequent words in a list of documents.

        Parameters
        -----------
        documents: List[str]
            A list of documents, where each document is a string representing a list of words.
        top_n: int, optional
            The number of most frequent words to return. Default is 10.

        Returns
        --------
        List[Tuple[str, int]]
            A list of tuples, where each tuple contains a word and its frequency, sorted in descending order of frequency.
        """
        word_freq = Counter()
        for doc in documents:
            word_freq.update(doc.split())

        return word_freq.most_common(top_n)

    def cluster_kmeans_WCSS(self, emb_vecs: List, n_clusters: int) -> Tuple[List, List, float]:
        """ This function performs K-means clustering on a list of input vectors and calculates the Within-Cluster Sum of Squares (WCSS) for the resulting clusters.

        This function implements the K-means algorithm and returns the cluster centroids, cluster assignments for each input vector, and the WCSS value.

        The WCSS is a measure of the compactness of the clustering, and it is calculated as the sum of squared distances between each data point and its assigned cluster centroid. A lower WCSS value indicates that the data points are closer to their respective cluster centroids, suggesting a more compact and well-defined clustering.

        The K-means algorithm works by iteratively updating the cluster centroids and reassigning data points to the closest centroid until convergence or a maximum number of iterations is reached. This function uses a random initialization of the centroids and runs the algorithm for a maximum of 100 iterations.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.
        n_clusters: int
            The number of clusters to form.

        Returns
        --------
        Tuple[List, List, float]
            Three elements:
            1) A list containing the cluster centers.
            2) A list containing the cluster index for each input vector.
            3) The Within-Cluster Sum of Squares (WCSS) value for the clustering.
        """
        cluster_centers, cluster_indices = self.cluster_kmeans(emb_vecs, n_clusters)
        wcss = 0
        for i in range(n_clusters):
            wcss += np.sum(np.linalg.norm(emb_vecs[cluster_indices == i] - cluster_centers[i], axis=1) ** 2)

        return cluster_centers, cluster_indices, wcss

    def cluster_hierarchical_single(self, emb_vecs: List) -> List:
        """
        Clusters input vectors using the hierarchical clustering method with single linkage.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        return AgglomerativeClustering(n_clusters=None, linkage='single', distance_threshold=0).fit_predict(emb_vecs)

    def cluster_hierarchical_complete(self, emb_vecs: List) -> List:
        """
        Clusters input vectors using the hierarchical clustering method with complete linkage.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        return AgglomerativeClustering(n_clusters=None, linkage='complete', distance_threshold=0).fit_predict(emb_vecs)

    def cluster_hierarchical_average(self, emb_vecs: List) -> List:
        """
        Clusters input vectors using the hierarchical clustering method with average linkage.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        return AgglomerativeClustering(n_clusters=None, linkage='average', distance_threshold=0).fit_predict(emb_vecs)

    def cluster_hierarchical_ward(self, emb_vecs: List) -> List:
        """
        Clusters input vectors using the hierarchical clustering method with Ward's method.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        return AgglomerativeClustering(n_clusters=None, linkage='ward', distance_threshold=0).fit_predict(emb_vecs)

    def visualize_kmeans_clustering_wandb(self, data, n_clusters, project_name, run_name):
        """ This function performs K-means clustering on the input data and visualizes the resulting clusters by logging a scatter plot to Weights & Biases (wandb).

        This function applies the K-means algorithm to the input data and generates a scatter plot where each data point is colored according to its assigned cluster.
        For visualization use convert_to_2d_tsne to make your scatter plot 2d and visualizable.
        The function performs the following steps:
        1. Initialize a new wandb run with the provided project and run names.
        2. Perform K-means clustering on the input data with the specified number of clusters.
        3. Obtain the cluster labels for each data point from the K-means model.
        4. Create a scatter plot of the data, coloring each point according to its cluster label.
        5. Log the scatter plot as an image to the wandb run, allowing visualization of the clustering results.
        6. Close the plot display window to conserve system resources (optional).

        Parameters
        -----------
        data: np.ndarray
            The input data to perform K-means clustering on.
        n_clusters: int
            The number of clusters to form during the K-means clustering process.
        project_name: str
            The name of the wandb project to log the clustering visualization.
        run_name: str
            The name of the wandb run to log the clustering visualization.

        Returns
        --------
        None
        """
        # Initialize wandb
        run = wandb.init(project=project_name, name=run_name)

        # Perform K-means clustering
        cluster_centers, cluster_indices = self.cluster_kmeans(data, n_clusters)
        # kmeans = KMeans(n_clusters=n_clusters)
        # cluster_indices = kmeans.fit_predict(data)
        # cluster_centers = kmeans.cluster_centers_

        # Plot the clusters
        plt.figure(figsize=(8, 8))
        scatter = plt.scatter(data[:, 0], data[:, 1], c=cluster_indices, cmap='viridis', alpha=0.5)
        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', s=100, marker='x')
        plt.colorbar(scatter)
        plt.title(f'K-means Clustering with {n_clusters} clusters')

        # Log the plot to wandb
        wandb.log({"kmeans_clustering": wandb.Image(plt)})

        # Close the plot display window if needed (optional)
        plt.close()

    def wandb_plot_hierarchical_clustering_dendrogram(self, data, project_name, linkage_method, run_name):
        """ This function performs hierarchical clustering on the provided data and generates a dendrogram plot, which is then logged to Weights & Biases (wandb).

        The dendrogram is a tree-like diagram that visualizes the hierarchical clustering process. It shows how the data points (or clusters) are progressively merged into larger clusters based on their similarity or distance.

        The function performs the following steps:
        1. Initialize a new wandb run with the provided project and run names.
        2. Perform hierarchical clustering on the input data using the specified linkage method.
        3. Create a linkage matrix, which represents the merging of clusters at each step of the hierarchical clustering process.
        4. Generate a dendrogram plot using the linkage matrix.
        5. Log the dendrogram plot as an image to the wandb run.
        6. Close the plot display window to conserve system resources.

        Parameters
        -----------
        data: np.ndarray
            The input data to perform hierarchical clustering on.
        linkage_method: str
            The linkage method for hierarchical clustering. It can be one of the following: "average", "ward", "complete", or "single".
        project_name: str
            The name of the wandb project to log the dendrogram plot.
        run_name: str
            The name of the wandb run to log the dendrogram plot.

        Returns
        --------
        None
        """
        run = wandb.init(project=project_name, name=run_name)
        # Perform hierarchical clustering
        # cluster_indices = AgglomerativeClustering(n_clusters=None, linkage=linkage_method, distance_threshold=0).fit_predict(data)
        Z = linkage(data, method=linkage_method)

        # Create linkage matrix for dendrogram
        plt.figure(figsize=(10, 7))
        dendrogram(Z)
        plt.title(f'Hierarchical Clustering Dendrogram - {linkage_method} linkage')
        plt.xlabel('Data Points')
        plt.ylabel('Distance')
        wandb.log({"hierarchical_clustering_dendrogram": wandb.Image(plt)})
        plt.close()

    def plot_kmeans_cluster_scores(self, embeddings: List, true_labels: List, k_values: List[int], project_name=None,
                                   run_name=None):
        """ This function, using implemented metrics in clustering_metrics, calculates and plots both purity scores and silhouette scores for various numbers of clusters.
        Then using wandb plots the respective scores (each with a different color) for each k value.

        Parameters
        -----------
        embeddings : List
            A list of vectors representing the data points.
        true_labels : List
            A list of ground truth labels for each data point.
        k_values : List[int]
            A list containing the various values of 'k' (number of clusters) for which the scores will be calculated.
            Default is range(2, 9), which means it will calculate scores for k values from 2 to 8.
        project_name : str
            Your wandb project name. If None, the plot will not be logged to wandb. Default is None.
        run_name : str
            Your wandb run name. If None, the plot will not be logged to wandb. Default is None.

        Returns
        --------
        None
        """
        metrics = ClusteringMetrics()
        silhouette_scores = []
        purity_scores = []
        # Calculating Silhouette Scores and Purity Scores for different values of k
        for k in k_values:
            # Using implemented metrics in clustering_metrics, get the score for each k in k-means clustering
            # and visualize it.
            kmeans = KMeans(n_clusters=k)
            cluster_indices = kmeans.fit_predict(embeddings)
            silhouette_scores.append(silhouette_score(embeddings, cluster_indices))
            purity_scores.append(metrics.purity_score(true_labels, cluster_indices))

        # Plotting the scores
        plt.figure(figsize=(8, 6))
        plt.plot(k_values, silhouette_scores, label='Silhouette Score')
        plt.plot(k_values, purity_scores, label='Purity Score')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Score')
        plt.title('K-means')
        plt.legend()

        # Logging the plot to wandb
        if project_name and run_name:
            import wandb
            run = wandb.init(project=project_name, name=run_name)
            wandb.log({"Cluster Scores": wandb.Image(plt)})
        # plt.show()

    def visualize_elbow_method_wcss(self, embeddings: List, k_values: List[int], project_name: str, run_name: str):
        """ This function implements the elbow method to determine the optimal number of clusters for K-means clustering based on the Within-Cluster Sum of Squares (WCSS).

        The elbow method is a heuristic used to determine the optimal number of clusters in K-means clustering. It involves plotting the WCSS values for different values of K (number of clusters) and finding the "elbow" point in the curve, where the marginal improvement in WCSS starts to diminish. This point is considered as the optimal number of clusters.

        The function performs the following steps:
        1. Iterate over the specified range of K values.
        2. For each K value, perform K-means clustering using the `cluster_kmeans_WCSS` function and store the resulting WCSS value.
        3. Create a line plot of WCSS values against the number of clusters (K).
        4. Log the plot to Weights & Biases (wandb) for visualization and tracking.

        Parameters
        -----------
        embeddings: List
            A list of vectors representing the data points to be clustered.
        k_values: List[int]
            A list of K values (number of clusters) to explore for the elbow method.
        project_name: str
            The name of the wandb project to log the elbow method plot.
        run_name: str
            The name of the wandb run to log the elbow method plot.

        Returns
        --------
        None
        """
        # Initialize wandb
        run = wandb.init(project=project_name, name=run_name)

        # Compute WCSS values for different K values
        wcss_values = []
        for k in k_values:
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(embeddings)
            wcss_values.append(kmeans.inertia_)

        # Plot the elbow method
        plt.figure(figsize=(8, 6))
        plt.plot(k_values, wcss_values)
        plt.title('Elbow Method')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('WCSS')

        # Log the plot to wandb
        wandb.log({"Elbow Method": wandb.Image(plt)})
        plt.close()

if __name__ == '__main__':
    cu = ClusteringUtils()
    data = np.concatenate((
        np.random.normal(loc=(0, 0), scale=(1, 1), size=(100, 2)),
        np.random.normal(loc=(10, 5), scale=(3, 1.5), size=(100, 2)),
        np.random.normal(loc=(-30, 40), scale=(6, 30), size=(100, 2))
    ))
    true_labels = np.concatenate((np.zeros(100), np.ones(100), np.full(100, 2)))
    
    cu.visualize_kmeans_clustering_wandb(data, 3, 'test', 'test')
    cu.plot_kmeans_cluster_scores(data, true_labels, range(2, 9), 'kmeans-scores', 'test')
    cu.visualize_elbow_method_wcss(data, [2, 3, 4, 5], 'elbow', 'test')
    cu.wandb_plot_hierarchical_clustering_dendrogram(data, 'test', 'average', 'average-link')
    cu.wandb_plot_hierarchical_clustering_dendrogram(data, 'test', 'ward', 'ward-link')
    cu.wandb_plot_hierarchical_clustering_dendrogram(data, 'test', 'complete', 'complete-link')
    cu.wandb_plot_hierarchical_clustering_dendrogram(data, 'test', 'single', 'single-link')
    print('Done!')
