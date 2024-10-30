from pyspark import SparkContext
import sys
import json
import time


# /opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit --executor-memory 4G --driver-memory 4G task2.py ../resource/asnlib/publicdata/review.json results.json 10
sctask_2 = SparkContext('local[*]','task2')
test_rev_json = sys.argv[1]
output_task2 = sys.argv[2]

# System default partitioning

start_time = time.time()

test_rev_rdd = sctask_2.textFile(test_rev_json)

test_review_rdd= test_rev_rdd.mapPartitions(lambda x:[json.loads(line) for line in x])

t10_business_reviews = test_review_rdd.map(lambda x:(x['business_id'],1))
t10_business_reviews_reduced = t10_business_reviews.reduceByKey(lambda a,b:a+b)
t10_business_reviews_sorted = t10_business_reviews.sortBy(lambda x:(-x[1],x[0])).take(10)

end_time = time.time()

exe_time = end_time - start_time

numPartition1 = t10_business_reviews_reduced.getNumPartitions()

partitioned_data = t10_business_reviews_reduced.glom()
partition_sizes = partitioned_data.map(len).collect()

# custom partition function

start_time_c = time.time()

test_rev_rdd2 = sctask_2.textFile(test_rev_json)

test_review_parsed2 = test_rev_rdd2.mapPartitions(lambda x:[json.loads(line) for line in x])

# convert each character in business id to ASCII int value and sum the values and take the value mod the system default number of partitions
n_partitions = int(sys.argv[3])
def char_partition(key):
    char_sum = sum(ord(char) for char in key)
    partition_number = char_sum % n_partitions
    return partition_number

t10_business_reviews_custom = test_review_parsed2.map(lambda x:(x['business_id'],1)).partitionBy(n_partitions,lambda key: char_partition(key))
t10_business_reviews_custom_red = t10_business_reviews_custom.reduceByKey(lambda a,b:a+b)
t10_business_reviews_custom_sort = t10_business_reviews_custom.sortBy(lambda x:(-x[1],x[0])).take(10)

end_time_c = time.time()
exe_time_c = end_time_c - start_time_c

numPartition_c = t10_business_reviews_custom_red.getNumPartitions()
partitioned_data_c = t10_business_reviews_custom_red.glom()
partition_sizes_c = partitioned_data_c.map(len).collect()
# results
output_results = {"default":{"n_partition":numPartition1,"n_items": partition_sizes,"exe_time":exe_time},"customized":{"n_partition":numPartition_c,"n_items": partition_sizes_c,"exe_time":exe_time_c}}
t2_json = json.dumps(output_results, indent=3)

with open(output_task2, "w") as answers:
    answers.write(t2_json)