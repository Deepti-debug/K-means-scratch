'''
Implement K-Means algorithm
'''
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from numpy import linalg as LA
from sklearn.metrics import silhouette_score


class KMeans:
    def __init__(self, features, k, max_iters):
        self.features = features
        self.k = k
        self.max_iters = max_iters
        rng = np.random.default_rng()
        #NOTE: Check if this initialization is correct
        self.centroids = rng.uniform(low=-1.0, high=1.0, size=(self.k, self.features.shape[1]))

    
    def compute_distance_matrix(self, pts1, pts2):
        '''
        Function to compute the euclidean distance between points pts1, pts2
        '''
        return (np.sum((pts1[None,:] - pts2[:,None])**2, axis=-1))


    def compute_wcss(self, distance_matrix, memberships):
        wcss = 0
        for cluster_iter in range(0, self.k):
            idxs = np.where(memberships == cluster_iter)
            wcss += np.sum(distance_matrix[idxs,cluster_iter])
        return wcss 
    
    def run(self):
        '''
        Main Function for running K-Means
        '''
        for i_iter in range(0, self.max_iters):
            distance_matrix = self.compute_distance_matrix(self.centroids, self.features)
            memberships = np.argmin(distance_matrix, axis=1)
            #Recalculate centroids
            for j_iter in range(0, self.k):
                idxs = np.where(memberships == j_iter)
                self.centroids[j_iter, :] = np.mean(self.features[idxs], axis=0)
            #Check for convergence
            if i_iter >= 1:
                deviation = LA.norm(centroids_old - self.centroids)
                if deviation <= 1e-4:
                    break 
            centroids_old = np.copy(self.centroids)
            
        wcss = self.compute_wcss(distance_matrix, memberships)
        # print(f"k:{self.k}, silhouette score:{silhuette_score(self.features, memberships)}")
        # silscore = silhouette_score(self.features, memberships)
        return wcss, None 


def main(data):
    k_ls = []
    wcss_ls = []
    silscore_ls = []
    for k in [2,3,5,7]:
        KMeans_obj = KMeans(data, k, max_iters=100)
        wcss, silscore = KMeans_obj.run()
        k_ls.append(k)
        wcss_ls.append(wcss)
        silscore_ls.append(silscore)

    plt.plot(k_ls, wcss_ls)
    plt.show()





if __name__ == '__main__':
    data = np.load(f"data.npy")
    main(data)    