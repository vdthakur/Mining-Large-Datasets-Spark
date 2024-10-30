from pyspark import SparkContext
import sys
import json

## /opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit --executor-memory 4G --driver-memory 4G task1.py ../resource/asnlib/publicdata/test_review.json results.json

sctask_1 = SparkContext('local[*]','task1')
test_rev_json = sys.argv[1]
answers_task1 = sys.argv[2]

test_rev_rdd =sctask_1.textFile(test_rev_json)
test_review_rdd = test_rev_rdd.mapPartitions(lambda x:[json.loads(review) for review in x])
# task1_a
review_count = test_review_rdd.count()
# task1_b
reviews_2018 = test_review_rdd.filter(lambda x:'2018' in x['date']).count()
# 1c
distinct_users = test_review_rdd.map(lambda x:x['user_id']).distinct().count()

# 1d
t10_largest_reviewers = test_review_rdd.map(lambda x:(x['user_id'],1)).reduceByKey(lambda a,b:a+b)
t10_largest_reviewers = t10_largest_reviewers.sortBy(lambda x: (-x[1],x[0])).take(10)

# 1e
distinct_business_reviews = test_review_rdd.map(lambda x:x['business_id']).distinct().count()

# 1f
t10_business_reviews = test_review_rdd.map(lambda x:(x['business_id'],1)).reduceByKey(lambda a,b:a+b)
t10_business_reviews = t10_business_reviews.sortBy(lambda x:(-x[1],x[0])).take(10)


all_results = {"n_review":review_count,"n_review_2018":reviews_2018,"n_user":distinct_users, "top10_user":t10_largest_reviewers,"n_business":distinct_business_reviews,"top10_business":t10_business_reviews}
compiled_json_results = json.dumps(all_results, indent=3)


with open(answers_task1, "w") as answers:
    answers.write(compiled_json_results)