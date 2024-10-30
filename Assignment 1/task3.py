from pyspark import SparkContext
import sys
import json
import time

# /opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit --executor-memory 4G --driver-memory 4G task3.py ../resource/asnlib/publicdata/test_review.json ../resource/asnlib/publicdata/business.json hi.txt results.json

sctask_3 = SparkContext('local[*]','task3')

test_rev_json = sys.argv[1]
business_json = sys.argv[2]
output_3a = sys.argv[3]
output_3b = sys.argv[4]

# load data for both RDD's

test_rev_rdd = sctask_3.textFile(test_rev_json)
business_rdd = sctask_3.textFile(business_json)

test_review_rdd_a = test_rev_rdd.mapPartitions(lambda x:[json.loads(line) for line in x])
test_review_values_a = test_review_rdd_a.map(lambda x: (x['business_id'], x["stars"]))

business_rdd_a = business_rdd.mapPartitions(lambda x:[json.loads(line) for line in x])
business_values_a = business_rdd_a.map(lambda x: (x['business_id'], x["city"]))

# join the two RDD's together

# looks like (business_id,(stars,city))
review_business_rdd_a = test_review_values_a.leftOuterJoin(business_values_a)


# part 3a

# looks like (city,(stars,1))
avg_stars_map_a = review_business_rdd_a.map(lambda x: (x[1][1],(x[1][0],1)))

# group based on city
# a[0] + b[0] is sum of stars from same city
# a[1] + b[1] is sum of reviews from city
avg_stars_red_a = avg_stars_map_a.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))

# x[0] denotes is sum of stars from the city
# x[1] denotes is sum of reviews from the city
avg_stars_rdd_a = avg_stars_red_a.mapValues(lambda x: x[0]/x[1]).sortBy(lambda x:(-x[1],x[0]))
avg_stars_output_a = avg_stars_rdd_a.collect()

with open(output_3a, "w") as answers:
        answers.write("city,stars\n")
        for city, avg_star in avg_stars_output_a:
            answers.write(f"{city},{avg_star}\n")

# part 3b

start_time_1b = time.time()

test_rev_rdd_1b = sctask_3.textFile(test_rev_json)
business_rdd_1b = sctask_3.textFile(business_json)

test_review_rdd_1b = test_rev_rdd_1b.mapPartitions(lambda x:[json.loads(line) for line in x])
test_review_values_1b = test_review_rdd_1b.map(lambda x: (x['business_id'], x["stars"]))


business_rdd_1b = business_rdd_1b.mapPartitions(lambda x:[json.loads(line) for line in x])
business_values_1b = business_rdd_1b.map(lambda x: (x['business_id'], x["city"]))

# looks like (business_id,(stars,city))
review_business_rdd_1b = test_review_values_1b.leftOuterJoin(business_values_1b)

# looks like (city,(stars,1))
avg_stars_map_1b = review_business_rdd_1b.map(lambda x: (x[1][1],(x[1][0], 1)))
avg_stars_red_1b = avg_stars_map_1b.reduceByKey(lambda a, b:(a[0] + b[0],a[1] + b[1]))
avg_stars_rdd_1b = avg_stars_red_1b.mapValues(lambda x: x[0]/x[1]).collect()
pyt_s = sorted(avg_stars_rdd_1b,key=lambda x: (-x[1], x[0]))
print(pyt_s[:10])

end_time_1b = time.time()

ex_time_1b = end_time_1b - start_time_1b

start_time_2b = time.time()

test_rev_rdd_2b = sctask_3.textFile(test_rev_json)
business_rdd_2b = sctask_3.textFile(business_json)

test_review_rdd_2b =  test_rev_rdd_2b.mapPartitions(lambda x:[json.loads(line) for line in x])
test_review_values_2b = test_review_rdd_2b.map(lambda x: (x['business_id'], x["stars"]))

business_rdd_2b = business_rdd_2b.mapPartitions(lambda x:[json.loads(line) for line in x])
business_values_2b = business_rdd_2b.map(lambda x: (x['business_id'], x["city"]))

review_business_rdd_2b = test_review_values_2b.leftOuterJoin(business_values_2b)

avg_stars_map_2b = review_business_rdd_2b.map(lambda x: (x[1][1],(x[1][0],1)))
avg_stars_red_2b = avg_stars_map_2b.reduceByKey(lambda a,b: (a[0] + b[0],a[1] + b[1]))
avg_stars_rdd_2b = avg_stars_red_2b.mapValues(lambda x: x[0]/x[1]).sortBy(lambda x:(-x[1],x[0])).take(10)

print(avg_stars_rdd_2b)

end_time_2b = time.time()
ex_time_2b = end_time_2b - start_time_2b

results = {"m1":ex_time_1b,"m2":ex_time_2b,"reason":"After running several times, it was noted that Python sort resulted in a quicker execution time than Spark sort did. On smaller datasets, this can be the case as there is no need for parallel/distributed processing when the data can fit in main memory. The transfer of the small amount of data done in the parallel/distributed processing likely resulted in the slower execution time for Spark. This trends however held consistent on the full dataset as well which is likely a result of how the data is being partioned. If the data is not partitioned efficiently, Spark's sort process can take longer as the shuffle requires for data to be transferred across the nodes."}
results_json = json.dumps(results,indent=3)

with open(output_3b, "w") as answers:
    answers.write(results_json)