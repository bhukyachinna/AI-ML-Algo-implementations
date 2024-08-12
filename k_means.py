"""
- This is the implementation of K-Means algorithm.
"""

import numpy as np

class KMeans:
    def __init__(self, n_clusters = 2, max_iter = 100):
        self.n_clusters = n_clusters     # The number of clusters to form as well as the number of centroids to generate.
        self.max_iter = max_iter         # Maximum number of iterations of the k-means algorithm for a single run.
        self.centroids = None
        self.cluster_centers_ = None     # Coordinates of cluster centers.
        self.labels_ = None              # Labels of each point.
        self.n_iter_ = None              # Number of iterations run.
        
    def __euclidean_distance(self, p1, p2):
        return np.linalg.norm(p1 - p2)
        
    def fit_predict(self, X):
        # Randomly generating centroids
        idx = np.random.randint(0, X.shape[0], self.n_clusters)
        self.centroids = X[idx]
        self.labels_ = np.ones(X.shape[0])
        self.n_iter_ = -1
        
        for self.n_iter_ in range(self.max_iter):
            # Assign clusters
            for i in range(X.shape[0]):
                distance = []
                for centroid in self.centroids:
                    distance.append(self.__euclidean_distance(X[i], centroid))
                self.labels_[i] = np.argmin(distance)
            
            # Move centroids
            old_centroids = np.copy(self.centroids)
            for j in range(self.n_clusters):
                cluster_j = X[self.labels_ == j]
                np.append(cluster_j, [self.centroids[j]], axis = 0)
                self.centroids[j] = np.mean(cluster_j, axis = 0) 
                
            # Chech finish condition
            if (old_centroids == self.centroids).all():
                break
        
        return self.labels_