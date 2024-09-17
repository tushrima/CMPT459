import numpy as np


class KMeans():

    def __init__(self, n_clusters: int, init: str='random', max_iter = 300):
        """

        :param n_clusters: number of clusters
        :param init: centroid initialization method. Should be either 'random' or 'kmeans++'
        :param max_iter: maximum number of iterations
        """
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.centroids = None # Initialized in initialize_centroids()

    def fit(self, X: np.ndarray):
        self.initialize_centroids(X)
        iteration = 0
        clustering = np.zeros(X.shape[0])
        while iteration < self.max_iter:
            # your code
            #gets X as input, finds n_clusters clusters and returns cluster labels in a Numpy array
            old_centroids = self.centroids.copy()
            #assigning clusters
            clustering = np.argmin(self.euclidean_distance(X, self.centroids), axis=1)
            self.update_centroids(clustering, X) ##pdate centroids
            #if convergence
            if np.all(old_centroids == self.centroids):
                break
            iteration += 1
        return clustering

    def update_centroids(self, clustering: np.ndarray, X: np.ndarray):
        #your code
        self.centroids = np.array([np.mean(X[clustering == i], axis=0) for i in range(self.n_clusters)])


    def initialize_centroids(self, X: np.ndarray):
        """
        Initialize centroids either randomly or using kmeans++ method of initialization.
        :param X:
        :return:
        """
        if self.init == 'random':
            # your code
            indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
            self.centroids = X[indices]
        elif self.init == 'kmeans++':
            # your code
            self.centroids = np.zeros((self.n_clusters, X.shape[1]))
            self.centroids[0] = X[np.random.choice(X.shape[0])]
            for i in range(1, self.n_clusters):
                dist = np.min(self.euclidean_distance(X, self.centroids[:i]), axis=1)
                probs = dist / np.sum(dist)
                index = np.random.choice(X.shape[0], p=probs)
                self.centroids[i] = X[index]
        else:
            raise ValueError('Centroid initialization method should either be "random" or "k-means++"')

    def euclidean_distance(self, X1:np.ndarray, X2:np.ndarray):
        """
        Computes the euclidean distance between all pairs (x,y) where x is a row in X1 and y is a row in X2.
        Tip: Using vectorized operations can hugely improve the efficiency here.
        :param X1:
        :param X2:
        :return: Returns a matrix `dist` where `dist_ij` is the distance between row i in X1 and row j in X2.
        """
        # your code
        #https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
        return np.linalg.norm(X1[:, np.newaxis] - X2, axis=2)

    def silhouette(self, clustering: np.ndarray, X: np.ndarray):
        # your code
        #silhouette coefficient is a measure of how similar a data point is within-cluster compared to other clusters.
        num_samples = X.shape[0]
        silhouette_vals = np.zeros(num_samples)

        for i in range(num_samples):
            cluster_label = clustering[i]
            cluster_points = X[clustering == cluster_label]
            a_i = np.mean(np.linalg.norm(cluster_points - X[i], axis=1))

            b_i = float('inf')
            for j in range(self.n_clusters):
                if j != cluster_label:
                    other_cluster_points = X[clustering == j]
                    b_ij = np.mean(np.linalg.norm(other_cluster_points - X[i], axis=1))
                    b_i = min(b_i, b_ij)
#(b - a) / max(a,b)
            silhouette_vals[i] = (b_i - a_i) / max(a_i, b_i)

        return np.mean(silhouette_vals)