from pyspark import SparkContext
import sys
import math
import random
import hashlib
from itertools import combinations
import csv
# import time

sctask_1 = SparkContext('local[*]','task1')
input = sys.argv[1]
output = sys.argv[2]


# start = time.time()
yelp_rdd =sctask_1.textFile(input)
first_line = yelp_rdd.first()
yelp_rdd = yelp_rdd.filter(lambda line: line != first_line).map(lambda row: row.strip().split(','))
yelp_rdd = yelp_rdd.map(lambda row: (row[1], row[0])).groupByKey().mapValues(set)  
all_users = yelp_rdd.flatMap(lambda row: row[1]).distinct().sortBy(lambda user: user).collect()

# create a dictionary of users and their index
indexed_users_rdd = yelp_rdd.flatMap(lambda row: row[1]).distinct().zipWithIndex()
user_index_dict = dict(indexed_users_rdd.collect())

n_bins = len(all_users)
# print(n_bins)
n_hash_functions = 170

# function to generate primes
def create_primes(start_prime,stop_prime):
    primes = []
    for possible_prime in range(start_prime, stop_prime):
        for i in range(2, int(math.sqrt(possible_prime)) + 1):
            if possible_prime % i == 0:
                break
        else:
            primes.append(possible_prime)
            if len(primes)== n_hash_functions + 1:
                break
    return primes

# generate the prime numbers
possible_prime_list = create_primes(n_bins,30000)
# retrieve the prime numbers (same amount of hash function)
prime_number = random.sample(possible_prime_list, n_hash_functions)
a_hash = random.sample(range(1, n_bins), n_hash_functions)
b_hash = random.sample(range(1, n_bins), n_hash_functions)

# function to create the hash functions
def hash_function(a, b, p, m):
    return lambda x: (a * x + b) % p % m

# create a list of hash functions using the random values previously generated
hash_functions = [hash_function(a_hash[i], b_hash[i], prime_number[i], n_bins) for i in range(n_hash_functions)]

# takes in the set of users, looks up their index and utilizing the corresponding hash function from the list of hash functions to create the minahsh signature
def minhash_signature(user_set):
    signature = [min([h(user_index_dict[user]) for user in user_set]) for h in hash_functions]
    return signature

# values of the yelp rdd are the sets of users so we map the values
minhash_rdd = yelp_rdd.mapValues(minhash_signature)

# defining the number of bands and rows per band
num_bands = 45
rows_per_band = 3

# function hash the bands using standard python library hashlib
def hash_band(band):
    return hashlib.md5(str(band).encode('utf-8')).hexdigest()

# create the bands and utilize the hash_band function to hash the bands 
def lsh(business_id, signature):
    bands = []
    # loop to divide the signature into `num_bands` bands
    for i in range(num_bands):
        # create a band by slicing the signature based on the number of rows per band
        band = signature[i * rows_per_band:(i + 1) * rows_per_band]
        # hash the created band
        band_hash = hash_band(band)
        # append (band hash, business_id) to the list
        bands.append((band_hash, business_id))
    return bands

# LSH function for each (business_id, signature) pair in the RDD
lsh_rdd = minhash_rdd.flatMap(lambda x: lsh(x[0], x[1])).groupByKey().mapValues(list)
# candidate pairs from the grouped business IDs by finding all unique combinations
candidate_pairs = lsh_rdd.flatMap(lambda x: combinations(x[1], 2)).distinct()

yelp_dict = yelp_rdd.collectAsMap()
yelp_broadcast = sctask_1.broadcast(yelp_dict)

# calculate Jaccard similarity
def jaccard_similarity(pair):
    business1, business2 = pair
    users1 = yelp_broadcast.value[business1]
    users2 = yelp_broadcast.value[business2]
    intersection = len(users1.intersection(users2))
    union = len(users1.union(users2))
    similarity = intersection / union
    return (pair, similarity)

# use spark to calculate the Jaccard similarity
jaccard_rdd = candidate_pairs.map(jaccard_similarity)
# filter pairs with Jaccard similarity >= 0.5
result_rdd = jaccard_rdd.filter(lambda x: x[1] >= 0.5)
# collect the filtered pairs
sorted_rdd = result_rdd.map(lambda pair: (tuple(sorted(pair[0])), pair[1]))

# sort the RDD by business_id_1, business_id_2, and Jaccard similarity
sorted_rdd = sorted_rdd.sortBy(lambda x: (x[0][0], x[0][1], x[1]))

# collect the sorted results
sorted_pairs = sorted_rdd.collect()

with open(output, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["business_id_1", "business_id_2", "similarity"])
    for pair in sorted_pairs:
        writer.writerow([pair[0][0],pair[0][1],pair[1]])

# end = time.time()
# print(end-start)