import numpy as np
from scipy import spatial

'''
Author: Antonio Lang
Date: 18 October 2022
'''

def K_Means(X,K,mu=None):
    if mu is None:
        # generates K random indicies to be the starting centers
        index_list = [*range(0,X.shape[0])]
        cluster_centers = np.empty((K, X.shape[1]))
        rand_index = np.random.choice(index_list, K, replace=False)
        cluster_centers = X[rand_index]
    cluster_assignment = []
    X = np.resize(X, (X.shape[0],cluster_centers.shape[1]))
    converged = False
    epoch = 0
    epoch_limit = 100

    while converged is not True and epoch < epoch_limit:
        distances = spatial.distance.cdist(cluster_centers, X) # generates the distances between samples and centers
        if epoch > 0: old_clusters = cluster_assignment
        cluster_assignment = assign_clusters(distances)
        if epoch > 0: converged = check_update(cluster_assignment, old_clusters)
        cluster_centers = calc_cluster_centers(cluster_assignment, X, K)
        epoch += 1
    return cluster_centers


def K_Means_better(X,K):
    converged_centers = False
    centers_list = []
    while converged_centers is not True:
        # runs K_means until a set of cluster centers is the majority
        centers_list.append(K_Means(X, K))
        converged_centers = check_converged_centers(centers_list)
    return centers_list[len(centers_list)-1]

def check_converged_centers(cluster_centers):
    unique_centers = []
    unique_center_count = []
    for centers in cluster_centers:
        # counts the number of unique cluster centers in the set of centers
        if centers not in unique_centers: unique_centers.append(centers), unique_center_count.append(1)
        else:
            for i, unique_center in enumerate(unique_centers):
                if unique_center == centers:
                    unique_center_count[i] += 1
    return max(unique_center_count) > len(cluster_centers) / 2 # returns True/False if a cluster is the majority

def assign_clusters(distances):
    assignments = []
    dist_by_data = np.transpose(distances)
    for sample in dist_by_data:
        # assigns samples to the closest cluster center
        assignments.append(np.argmin(sample))
    return assignments

def check_update(clusters_new, clusters_old):
    return clusters_new == clusters_old

def calc_cluster_centers(cluster_assignments, X, K):
    # iterates over the samples and checks their assignments
    num_samples = np.zeros(K)
    cluster_sums = [0] * K
    cluster_centers = np.zeros((K,len(X[0])))
    for i, sample in enumerate(X):
        cluster_sums[cluster_assignments[i]] += sample
        num_samples[cluster_assignments[i]] += 1
    for i in range(K):
        # takes the avereage value of the points as the cluster center
        cluster_centers[i] = cluster_sums[i] / num_samples[i]
    return cluster_centers