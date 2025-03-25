from pyspark import SparkContext
import sys
from collections import defaultdict
import time
import csv
from itertools import combinations
import shutil


sctask_2 = SparkContext('local[*]','task2')
filter_threshold = int(sys.argv[1])
support = int(sys.argv[2])
inp_path = sys.argv[3]
out_path = sys.argv[4]

# preprocessing to get rid of leading zeros and ensure values are ints
with open(inp_path, 'r') as input_file, open("edited_tafeng_data.csv", 'w', newline='') as output_file:
    reader = csv.reader(input_file)
    writer = csv.writer(output_file)
    header = next(reader)
    writer.writerow(header)
    for row in reader:
        row[5] = int(float(row[5]))  
        writer.writerow(row)


input_process_rdd = sctask_2.textFile("edited_tafeng_data.csv")
first_line = input_process_rdd.first()
input_rdd_no_h = input_process_rdd.filter(lambda line: line != first_line)
input_rdd_no_h = input_rdd_no_h.map(lambda line: line.split(","))
# pre processing to create new dataset
process_rdd = input_rdd_no_h.map(lambda row: (f"{row[0]}-{row[1]}", row[5]))
process_rdd = process_rdd.map(lambda row: f"{row[0]},{row[1]}")

newheader_rdd = sctask_2.parallelize(["DATE_CUSTOMER_ID,PRODUCT_ID"])
final_rdd = newheader_rdd.union(process_rdd)
final_rdd.coalesce(1).saveAsTextFile("preprocessed_data")
shutil.move("preprocessed_data/part-00000", "preprocessed_data.csv")
shutil.rmtree("preprocessed_data")
# new dataset createdd

start_time = time.time()
preprocessed_input_rdd = sctask_2.textFile("preprocessed_data.csv")

# splitting and stripping the inputted CSV file
baskets_rdd = preprocessed_input_rdd.map(lambda date_cust_id: date_cust_id.strip().split(","))
# map the first column to be date-customer id, and map the second column to be basket as a set (of product ids), convert to strings
baskets_rdd = baskets_rdd.map(lambda x: (str(x[0]), {str(x[1])}))
# combine the sets (values) of the same keys to create user basekts
in_rdd = baskets_rdd.reduceByKey(lambda a, b: a.union(b))
# only report the baskets who have a length greater than that of the filter threshold
filtered_rdd = in_rdd.filter(lambda x: len(x[1]) > filter_threshold).values()
partition_count = filtered_rdd.getNumPartitions()
# define the local support 
supp_local = support // partition_count


def find_frequent_singles(partition, supp_local):
    # convert the values in the partitions to a list
    # this list will contain multiple baskets that are found within each partition being mapped
    partition_data = list(partition)
    # dictionary creation to store the frequency (as values) of the single items
    item_counts = defaultdict(int)
    # for each basket of data found in the partition
    for basket in partition_data:
        for item in basket:
            # for every item found in the basket, increment by one
            item_counts[item] += 1
    # filter out the frequent singletons who meet the local supp threshold
    frequent_items = {item for item, count in item_counts.items() if count >= supp_local}
    # create a frozenset of each item that meets the local support threshold
    frequent_values = {frozenset([item]) for item in frequent_items}
    
    # return the partition data as a list and the frequent singles that were found
    return partition_data, frequent_values


def find_frequent_itemsets_partition(partition_data, frequent_values, supp_local):
    
    local_frequent_itemsets = frequent_values.copy()
    # begin the creation of itemset with size 2
    itemset_size= 2
    # while frequent sets are being produced (as long as there are more partitions)
    while frequent_values:
        # create a set to store the candidate values
        candidates = set()
        frequent_values_list = list(frequent_values)
        len_frequent_val = len(frequent_values_list)
        # for the index position of each frequent value
        for index in range(len_frequent_val):
            itemset1 = frequent_values_list[index]
            for next_index in range(index + 1, len_frequent_val):
                itemset2 = frequent_values_list[next_index]
                # check if both itemsets (excluding the last element) share the same prefix 
                if sorted(itemset1)[:-1] == sorted(itemset2)[:-1]:
                    # if so we can create new candidate sets from them
                    cand_set = itemset1 | itemset2
                    # add them to the set of candidates
                    if len(cand_set) == itemset_size:
                        candidates.add(cand_set)
        # prune the created candidate sets to remove any that have infrequent subsets
        pruned_candidates = set()
        # for each candidate set in the candidate sets
        for candidate in candidates:
            # create a subset of the candidates with a set size one less than the current itemset size
            subsets = combinations(candidate, itemset_size - 1)
            # check that all of these created smaller subsets can be found in the frequent subsets found in prior interations
            if all(frozenset(subset) in frequent_values for subset in subsets):
                # if so, that means the candidate set can be passed on as all of its subsets are also frequent
                pruned_candidates.add(candidate)
        # create a dictionary to track the frequency count of the candidate pairs in the original partition data
        candidate_counts = defaultdict(int)
        # for each basket of data found in the partition data
        for basket in partition_data:
            # for each candidate set that was passed on by the pruned candidates flow
            for candidate_set in pruned_candidates:
                # if the candidate set can be found in the basket set
                if candidate_set.issubset(basket):
                    # then increase the candidate count by 1
                    candidate_counts[candidate_set] += 1
        # assigning frequent values to the candidate sets that meet the local support, for as long as there are more, if not this while loop breaks
        frequent_values = {itemset for itemset, count in candidate_counts.items() if count >= supp_local}
        # updating the set for all frequent itemsets wih the new candidate sets that meet the local support 
        local_frequent_itemsets.update(frequent_values)
        # increase the itemset_size by 1 for the next iteration of increased set size
        itemset_size += 1
    # return the set of all frequent itemsets found in the partition by apriori
    return local_frequent_itemsets  


def apriori_local_partition(partition, supp_local):
    # find frequent singles
    partition_data, frequent_singles = find_frequent_singles(partition, supp_local)
    # find frequent itemsets of size 2 or greater
    local_frequent_itemsets = find_frequent_itemsets_partition(partition_data, frequent_singles, supp_local)
    # return the candidate frequent itemsets found from each partition
    return local_frequent_itemsets

# needs to be distinct because itemsets may appear frequent across multiple partitions and we need to account for that
partition_apriori = filtered_rdd.mapPartitions(lambda partition: apriori_local_partition(partition,supp_local)).distinct().collect()
# print(partition_apriori)


def find_frequent_all(partition, intermediate_candidates):
    # create a dictionary to track the frequency of the candidate sets found in each partition across partitions
    frequencies_cand = defaultdict(int)
    # check if any of the intermediate candidate sets appear in each partition
    for item in partition:
        candidates = filter(lambda candidate: candidate.issubset(item), intermediate_candidates)
        for candidate_set in candidates:
            frequencies_cand[candidate_set] += 1
    # return the candidate sets and their frequencies as a list
    frequent_list = [(candidate_set, freq) for candidate_set, freq in frequencies_cand.items()]
    return frequent_list


# this will map each partition and execute the find_frequent_all function which uses the partition and compares with sets made my apriori function
all_partitions_frequent_items = filtered_rdd.mapPartitions(lambda partition:find_frequent_all(partition, partition_apriori))
# print(all_partition_frequent_items.collect())
all_partitions_frequent_items  = all_partitions_frequent_items.reduceByKey(lambda x, y: x + y).filter(lambda x: x[1] >= support).map(lambda x: x[0]).collect()
# print(all_partition_frequent_items)

# write the output files
with open(out_path, 'w') as file:
    file.write('Candidates:\n')
    # creating a dict to group by item-set size
    grouped_candidates = defaultdict(list)
    # for each set produced by the local apriori partition function, append the set to its corresponding length
    for set in partition_apriori:
        grouped_candidates[len(set)].append(set)

    # sort the created dictionary
    # for each itemset length in the dictionary
    for length in sorted(grouped_candidates):
        # sort itemsets by lengt
        items = sorted(grouped_candidates[length], key=lambda x: sorted(x))
        # formatted as per guidelines given in instructions
        # sort lexico
        formatted_items = ["({})".format(', '.join("'{}'".format(x) for x in sorted(item))) for item in items]
        file.write(','.join(formatted_items) + '\n\n')


with open(out_path, 'a') as file:
    file.write('Frequent Itemsets:\n')
    grouped_frequent_items = defaultdict(list)
    for item in all_partitions_frequent_items:
        grouped_frequent_items[len(item)].append(item)

    for length in sorted(grouped_frequent_items):
        # sort itemsets
        items = sorted(grouped_frequent_items[length], key=lambda x: sorted(x))
        # formatted as per guidelines given in instructions
        # sort lexico
        formatted_items = ["({})".format(', '.join("'{}'".format(x) for x in sorted(item))) for item in items]
        file.write(','.join(formatted_items) + '\n\n')


end_time = time.time()
total = end_time - start_time
sctask_2.stop()
print(f'Duration: {total}')
