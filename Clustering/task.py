import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
import math
import sys
import time

# input arguments
input_f = sys.argv[1]
output_f = sys.argv[3]
num_clusters = int(sys.argv[2])

# start = time.time()
# load the data using numpy
data = np.loadtxt(input_f,delimiter=",")
# define the number of points and # of dimensions (features)
num_points,dimensions = data[:,2:].shape
# define the merge closeness threshold to be 2 * sqrt(dimensions)
distance_thresh = 2* math.sqrt(dimensions)
# randomly shuffle the data
np.random.shuffle(data)
# split the data into 5 chunks so each chunk has 20% of the data
num_chunks = 5
chunks = np.array_split(data, num_chunks)

# function used to calculate the mahalanobis distance
def mahalanobis_distance(point,cluster_stats):
    # define the number of points, sum of points, and sum of squares of points using the cluster stats that are already generated
    n = cluster_stats['N']
    sum = cluster_stats['SUM']
    sumsq = cluster_stats['SUMSQ']
    # calculate the centroid 
    centroid = sum/n
    # caculate the variance using formula in slides
    variance = (sumsq / n) -(centroid ** 2)
    # avoid dividing by zero in the chance the variance is 0
    variance = np.where(variance == 0,1e-10,variance) 
    # calculate the difference between the point and the centroid
    diff = point-centroid
    # return the mahalanobis distance using the formula
    mb_d = np.sqrt(np.sum((diff / np.sqrt(variance))**2))
    return mb_d

# write /create the output file
with open(output_f, "w") as f:
    f.write("The intermediate results:\n")

# function used to write the intermediate results to the output file through each iteration (5 total)
def intermediate_results(output_f,round_number,ds_points,compressed_clusters,cs_points,rs_points):
    with open(output_f, "a") as f:
        f.write(f"Round {round_number}: {ds_points},{compressed_clusters},{cs_points},{rs_points}\n")

# define the discard set clusters, compression set clusters, and retained set
discard_clusters = {}
compressed_clusters = {}
RS = []
# list to store the indices of the retained set
retained_set_indices = []
# dictionary to store the cluster assignments
cluster_assignments = {}

# for each randomly generated 20% chunk of the data
for chunk_index,chunk in enumerate(chunks):
    # retrieve the indices and features from the chunk
    chunk_indices = chunk[:,0].astype(int)
    chunk_data = chunk[:,2:]
    # if this is the first chunk initialize the clustering process
    if chunk_index == 0:
        # define a large k (eg: 5 times the number of clusters)
        k_large = num_clusters *5
        # run kmeans on the chunk data with k = k_large
        kmeans = KMeans(n_clusters=k_large,random_state=0).fit(chunk_data)
        # retrieve the labels from the kmeans clustering
        labels = kmeans.labels_
        # create a dictionary to store the cluster counts
        cluster_counts = defaultdict(int)
        # create a dictionary to store the clusters to points
        clusters_to_points = defaultdict(list)
        # for each index and label in the labels
        for index, label in enumerate(labels):
            # increase the cluster count for the specific label
            cluster_counts[label] += 1
            # append the index of the points to the cluster label in the cluster to points dictionary
            clusters_to_points[label].append(index)
        # create a list to store the index of points in retained set
        retained_set_indices= []
        # create a list to store the points in retained set
        RS = []
        # for each cluster id and count in the cluster counts
        for cluster_id,count in cluster_counts.items():
            # if the count is 1, then the cluster goes to the retained set
            if count == 1:
                # retrieve the index of the point
                index = clusters_to_points[cluster_id][0]
                # append point to the retained set
                RS.append(chunk_data[index])
                # append the index of the point to the retained set indices
                retained_set_indices.append(chunk_indices[index])
        # define the remaining indices as the points that were not is_assigned to the retained set
        remaining_indices = [index for index in range(len(chunk_data)) if index not in clusters_to_points[cluster_id] or cluster_counts[labels[index]] > 1]
        # define the remaining data as the points that were not is_assigned to the retained set
        remaining_data = chunk_data[remaining_indices]
        # define the remaining global indices as the indices of the points that were not is_assigned to the retained set
        remaining_global_indices = chunk_indices[remaining_indices]
        # run kmeans on the remaining data with k = number of clusters
        kmeans = KMeans(n_clusters=num_clusters,random_state=0).fit(remaining_data)
        # retrieve the labels from the kmeans clustering
        labels = kmeans.labels_
        # for each cluster id in the number of clusters
        for cluster_id in range(num_clusters):
            # retrieve the data points that belonging to the cluster
            cluster_points = remaining_data[labels == cluster_id]
            # retrieve the global indices of the points that belong to the cluster
            cluster_global_indices = remaining_global_indices[labels == cluster_id]
            # add the cluster to the discard set clusters
            # calculate the sum of the points, sum of squares of the points, and number of points in the cluster
            discard_clusters[cluster_id] = {'N': len(cluster_points),'SUM':np.sum(cluster_points, axis=0),'SUMSQ':np.sum(cluster_points ** 2, axis=0),'indices':cluster_global_indices.tolist(),}
        # if the retained set is greater than or equal to 5 times the number of clusters
        if len(RS) >= num_clusters * 5:
            # convert the retained set to an array
            rs_kmeans = np.array(RS)
            # run kmeans on the retained set with k = 5 times the number of clusters
            kmeans = KMeans(n_clusters=num_clusters * 5,random_state=0).fit(rs_kmeans)
            # retrieve the labels from the kmeans clustering
            labels = kmeans.labels_
            # create a dictionary to store the cluster counts
            cluster_counts = defaultdict(int)
            # create a dictionary to store the clusters to points
            clusters_to_points = defaultdict(list)
            # for each index and label in the labels
            for index, label in enumerate(labels):
                # increase the cluster count for the specific label
                cluster_counts[label] += 1
                # append the index of the points to the cluster label in the cluster to points dictionary
                clusters_to_points[label].append(index)

            # create a new list to store the points in retained set
            new_RS = []
            # create a new list to store the index of the in retained set
            new_retained_set_indices = []
            # for each cluster id and count in the cluster counts
            for cluster_id, count in cluster_counts.items():
                # if the count is 1, then the cluster goes to the retained set
                if count == 1:
                    # retrieve the index of the point
                    index = clusters_to_points[cluster_id][0]
                    # append point to the new retained set
                    new_RS.append(RS[index])
                    # append the index of the point to the new retained set indices
                    new_retained_set_indices.append(retained_set_indices[index])
                else:
                    # otherwise if count is greater, add the cluster to the compression set clusters
                    cluster_points = rs_kmeans[clusters_to_points[cluster_id]]
                    # define the indices of the points in the clusters of the compression set
                    cs_indices = [retained_set_indices[index] for index in clusters_to_points[cluster_id]]
                    # define the compression set using the total number of points, sum of points, sum of squares of points, and the indices of the points
                    compressed_clusters[cluster_id] = {'N':len(cluster_points),'SUM':np.sum(cluster_points, axis=0),'SUMSQ':np.sum(cluster_points ** 2, axis=0),'indices':cs_indices,}
            # RS is now the new retained set
            RS = new_RS
            # RS indices is now the new retained set indices
            retained_set_indices = new_retained_set_indices
    # if the length of the RS is not sufficient for kmeans clustering
    else: 
        # for each point and index in the next chunk of data and indices
        for point, index in zip(chunk_data,chunk_indices):
            # define the is_assigned marker as False
            is_assigned = False
            # the minimum distance is set to infinity as this will be updated
            min_distance = float('inf')
            # the closest cluster id is set to None as this will be updated
            closest_cluster_id = None
            # for each cluster id and stats in the discard set clusters
            for cluster_id, stats in discard_clusters.items():
                # calculate the mahalanobis distance using the point and the stats
                distance = mahalanobis_distance(point,stats)
                # if the distance is less than the closeness threshold (2*sqrt(dimensions) and less than the minimum distance (will be on first iteration due to infinity)
                if distance < distance_thresh and distance < min_distance:
                    # update the minimum distance
                    min_distance = distance
                    # update the closest cluster id
                    closest_cluster_id = cluster_id
            # if the minimum distance is less than the closeness threshold and the closest cluster id is not None
            if min_distance < distance_thresh:
                # update the stats of the closest cluster id
                stats = discard_clusters[closest_cluster_id]
                # increase the number of points in the cluster
                stats['N'] += 1
                # update the sum of the points
                stats['SUM'] += point
                # update the sum of the squares of the points
                stats['SUMSQ'] += point ** 2
                # append the index of the point to the cluster indices
                stats['indices'].append(index)
                # the point has been assigned
                is_assigned = True
            # otherwise if the minimum distance is greater than the closeness threshold
            else:
                # define the minimum distance as infinity
                min_distance = float('inf')
                # define the closest cluster id as None
                closest_cluster_id = None
                # for each cluster id and stats in the compression set clusters
                for cluster_id, stats in compressed_clusters.items():
                    # calculate the mahalanobis distance using the point and the stats
                    distance = mahalanobis_distance(point,stats)
                    # if the distance is less than the closeness threshold (2*sqrt(dimensions) and less than the minimum distance (will be on first iteration due to infinity)
                    if distance < distance_thresh and distance < min_distance:
                        # update the minimum distance
                        min_distance = distance
                        # update the closest cluster id
                        closest_cluster_id = cluster_id
                # if the minimum distance is less than the closeness threshold and the closest cluster id is not None
                if min_distance < distance_thresh:
                    # update the stats of the closest cluster id
                    stats = compressed_clusters[closest_cluster_id]
                    # increase the number of points in the cluster
                    stats['N'] += 1
                    # update the sum of the points
                    stats['SUM'] += point
                    # update the sum of the squares of the points
                    stats['SUMSQ'] += point **2
                    # append the index of the point to the cluster indices
                    stats['indices'].append(index)
                    # the point has been is_assigned
                    is_assigned = True
                # if the minimum distance is greater than the closeness threshold
                else:
                    # append the point to the retained set
                    RS.append(point)
                    # append the index of the point to the retained set indices
                    retained_set_indices.append(index)
        # if the length of the retained set is greater than or equal to 5 times the number of clusters
        if len(RS) >= num_clusters * 5:
            # convert the retained set to an array
            rs_kmeans = np.array(RS)
            # run kmeans on the retained set with k = 5 times the number of clusters
            kmeans = KMeans(n_clusters=num_clusters * 5,random_state=0).fit(rs_kmeans)
            # retrieve the labels from the kmeans clustering
            labels = kmeans.labels_
            # create a dictionary to store the cluster counts
            cluster_counts = defaultdict(int)
            # create a dictionary to store the clusters to points
            clusters_to_points = defaultdict(list)
            # for each index and label in the labels
            for index, label in enumerate(labels):
                # increase the cluster count for the specific label
                cluster_counts[label] += 1
                # append the index of the points to the cluster label in the cluster to points dictionary
                clusters_to_points[label].append(index)
            # define a new list to store the points in the retained set
            new_RS = []
            # define a new list to store the index of the points in the retained set
            new_retained_set_indices = []
            # for each cluster id and count in the cluster counts
            for cluster_id, count in cluster_counts.items():
                # if the count is 1, then the cluster goes to the retained set
                if count == 1:
                    # retrieve the index of the point
                    index = clusters_to_points[cluster_id][0]
                    # append point to the new retained set
                    new_RS.append(RS[index])
                    # append the index of the point to the new retained set indices
                    new_retained_set_indices.append(retained_set_indices[index])
                # otherwise if count is greater, add the cluster to the compression set clusters
                else:
                    # retrieve the points in the cluster
                    cluster_points = rs_kmeans[clusters_to_points[cluster_id]]
                    # retrieve the indices of the points in the cluster and store as compress set indices
                    cs_indices = [retained_set_indices[index] for index in clusters_to_points[cluster_id]]
                    # define unique key for stats in the compression set
                    cs_key = f"{cluster_id}"
                    # define the compression set using the total number of points, sum of points, sum of squares of points, and the indices of the points
                    compressed_clusters[cs_key] = {'N':len(cluster_points),'SUM':np.sum(cluster_points, axis=0),'SUMSQ': np.sum(cluster_points **2, axis=0),'indices':cs_indices,}
            # RS is now the new retained set
            RS = new_RS
            # RS indices is now the new retained set indices
            retained_set_indices = new_retained_set_indices

        # set is_merged status to True to begin the merging process
        is_merged = True
        # while is_merged is True
        while is_merged:
            # immediately set is_merged to False
            is_merged = False
            # for each cluster in he compression set clusters retrieve the key (chunk id and cluster id)
            cs_ids = list(compressed_clusters.keys())
            # for each cluster id in the compression set clusters
            for i in range(len(cs_ids)):
                # if is_merged is True, break out of the loop
                if is_merged:
                    break
                # retrieve the cluster id
                id1 = cs_ids[i]
                # for each next cluster id in the compression set clusters
                for j in range(i + 1,len(cs_ids)):
                    # retrieve the next cluster id
                    id2 = cs_ids[j]
                    # retrieve the relevant stats for the cluster id for the firs cluster
                    centroid1 = compressed_clusters[id1]['SUM']/compressed_clusters[id1]['N']
                    # retrieve the relevant stats for the cluster id for the second cluster
                    centroid2 = compressed_clusters[id2]['SUM']/compressed_clusters[id2]['N']
                    # calculate the distance between the two clusters using the centroids
                    distance = np.sqrt(np.sum(((centroid1 - centroid2) / (np.sqrt((compressed_clusters[id1]['SUMSQ'] / compressed_clusters[id1]['N']) - (centroid1 ** 2)) + 1e-10)) ** 2))
                    # if the distance is less than the closeness threshold (2*sqrt(dimensions))
                    if distance < distance_thresh:
                        # retrieve the stats for the first cluster
                        stats1 = compressed_clusters[id1]
                        # retrieve the stats for the second cluster
                        stats2 = compressed_clusters[id2]
                        # update the stats for the first cluster
                        stats1['N'] += stats2['N']
                        stats1['SUM'] += stats2['SUM']
                        stats1['SUMSQ'] += stats2['SUMSQ']
                        stats1['indices'].extend(stats2['indices'])
                        # delete the second cluster from the compression set clusters
                        del compressed_clusters[id2]
                        # set is_merged to True
                        is_merged = True
                        # break out of the loop
                        break

    # define the number of discard points through each iteration
    num_discard_points = sum([stats['N'] for stats in discard_clusters.values()])
    # define the number of compression set clusters through each iteration
    num_compressed_clusters = len(compressed_clusters)
    # define the number of compression points through each iteration
    num_compression_points = sum([stats['N'] for stats in compressed_clusters.values()])
    # define the number of retained points through each iteration
    num_retained_points = len(RS)
    # write the intermediate results to the output file
    intermediate_results(output_f, chunk_index + 1,num_discard_points,num_compressed_clusters,num_compression_points,num_retained_points)

# for each cluster in the compression set clusters
for cs_id, stats in list(compressed_clusters.items()):
    # set the minimum distance to infinity
    min_distance = float('inf')
    # set the closest cluster id to None
    closest_ds_id = None
    # for each cluster in the compression set clusters calculate the centroid
    centroid_cs = stats['SUM']/stats['N']
    # for each cluster in the discard set clusters
    for ds_id, ds_stats in discard_clusters.items():
        # for each cluster in the discard set clusters calculate the centroid
        centroid_ds = ds_stats['SUM']/ds_stats['N']
        # calculate the distance between the two clusters using the centroids
        distance = np.sqrt(np.sum(((centroid_cs - centroid_ds) / (np.sqrt((ds_stats['SUMSQ']/ds_stats['N']) - (centroid_ds **2)) + 1e-10)) **2))
        # if the distance is less than the closeness threshold (2*sqrt(dimensions)) and less than the minimum distance
        if distance < distance_thresh and distance < min_distance:
            # update the minimum distance
            min_distance = distance
            # update the closest cluster id
            closest_ds_id = ds_id
    # if closest_ds_id is defined
    if closest_ds_id is not None:
        # update the stats for the closest cluster id
        ds_stats = discard_clusters[closest_ds_id]
        ds_stats['N'] += stats['N']
        ds_stats['SUM'] += stats['SUM']
        ds_stats['SUMSQ'] += stats['SUMSQ']
        ds_stats['indices'].extend(stats['indices'])
        # delete the cluster from the compression set clusters
        del compressed_clusters[cs_id]

# for each cluster in the discard set clusters
for cluster_id, stats in discard_clusters.items():
    # for each index in the cluster indices
    for index in stats['indices']:
        # assign the cluster id to the index
        cluster_assignments[index] = cluster_id

# for each cluster in the compression set clusters
for cluster_id, stats in compressed_clusters.items():
    # for each index in the cluster indices
    for index in stats['indices']:
        # assign the cluster id to the index
        cluster_assignments[index] = -1 

# for each index in the retained set indices 
for index in retained_set_indices:
    # assign the cluster if as -1 to denote outliers to the index
    cluster_assignments[index] = -1  

# write the clustering results to the output file
with open(output_f, "a") as f:
    f.write("\nThe clustering results:\n")
    for index in sorted(cluster_assignments.keys()):
        f.write(f"{index},{cluster_assignments[index]}\n")

# end = time.time()
# duration = end - start
# print(f"Duration: {duration}")