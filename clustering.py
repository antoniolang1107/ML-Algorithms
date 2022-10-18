from random import sample
import numpy as np
from scipy import spatial

'''
Author: Antonio Lang
Date: 17 October 2022
'''

def K_Means(X,K,mu=None):
    if mu is None:
        cluster_centers = np.random.choice(X, K, replace=False)
    cluster_assignment = []
    cluster_centers = np.array([[5], [10]])
    X = np.resize(X, (X.shape[0],cluster_centers.shape[1]))
    converged = False
    epoch = 0
    epoch_limit = 10

    while converged is not True and epoch < epoch_limit:
        distances = spatial.distance.cdist(cluster_centers, X)
        if epoch > 0: old_clusters = cluster_assignment
        cluster_assignment = assign_clusters(distances)
        if epoch > 0: converged = check_update(cluster_assignment, old_clusters)
        cluster_centers = calc_cluster_centers(cluster_assignment, X, K)
        epoch += 1

    return cluster_centers


def K_Means_better(X,K):
    # run K_Means until receiving the same cluster centers
    
    pass

def assign_clusters(distances):
    assignments = []
    dist_by_data = np.transpose(distances)
    for sample in dist_by_data:
        assignments.append(np.argmin(sample))
    return assignments

def check_update(clusters_new, clusters_old):
    return clusters_new == clusters_old

def calc_cluster_centers(cluster_assignments, X, K):
    num_samples = np.zeros(K)
    cluster_sums = [0] * K
    cluster_centers = np.zeros((K,len(X[0])))
    for i, sample in enumerate(X):
        cluster_sums[cluster_assignments[i]] += sample
        num_samples[cluster_assignments[i]] += 1
    for i in range(K):
        cluster_centers[i] = cluster_sums[i] / num_samples[i]
    return cluster_centers