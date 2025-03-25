import os
os.environ["PYSPARK_SUBMIT_ARGS"] = "--packages graphframes:graphframes:0.8.2-spark3.1-s_2.12 pyspark-shell"
from pyspark.sql import SparkSession
import sys
from graphframes import GraphFrame

# spark-submit --packages graphframes:graphframes:0.8.2-spark3.1-s_2.12 task1.py <filter threshold><inputfile__path> <community_outputfile__path>

filter_threshold = int(sys.argv[1]) 
input_f = sys.argv[2]
community_output = sys.argv[3]

spark_t1 = SparkSession.builder.appName("task1").getOrCreate()

# read and process the data
ub_data_rdd = spark_t1.sparkContext.textFile(input_f)
first_line  = ub_data_rdd.first()

# remove header and split the data
cleaned_ub_data_rdd = ub_data_rdd.filter(lambda row:row != first_line).map(lambda row: row.strip().split(','))

# create (user, business) pairs
user_business_rdd = cleaned_ub_data_rdd.map(lambda x:(x[0], x[1]))

# create (business, users) pairs
business_user_rdd = user_business_rdd.map(lambda x:(x[1], x[0])).groupByKey().mapValues(set)

# create user pairs for each business
user_pairs = business_user_rdd.flatMap(lambda x:[(u1, u2) for u1 in x[1] for u2 in x[1] if u1 != u2])

# count numb of times each user pair appears
user_pair_counts = user_pairs.map(lambda x: ((x[0], x[1]), 1)).reduceByKey(lambda x, y: x + y)

# filter pairs that appear more than filter_threshold, defined as system argument
filtered_pairs = user_pair_counts.filter(lambda x: x[1] >= filter_threshold)

# create the dataframe for the edges
edges_df = spark_t1.createDataFrame(filtered_pairs.map(lambda x:(x[0][0], x[0][1])), ["src", "dst"])

# create vertices dataframe with vertices of edges post filtering
vertices_rdd = filtered_pairs.flatMap(lambda x: [x[0][0], x[0][1]]).distinct().map(lambda x: (x,))
vertices_df = spark_t1.createDataFrame(vertices_rdd, ["id"])

# use grahframes to create the graph using the vertices and edges as arguments
graphframes_graph_ub = GraphFrame(vertices_df,edges_df)

# find communities using label propagation algorithm
created_communities = graphframes_graph_ub.labelPropagation(maxIter=5)
# print(created_communities.show())

# create rdd from the communities and map the communities to (label, id) pairs
community_groups_rdd = created_communities.rdd
community_groups = community_groups_rdd.map(lambda row:(row['label'], row['id']))
# print(community_groups.take(5))

# group the communities by label and sort the users in each community
community_groups = community_groups.groupByKey().mapValues(lambda users: sorted(users))

# map the communities to (size, id, users) pairs joining the users with a comma
community_groups = community_groups.map(lambda x: (len(x[1]), x[1][0], ', '.join([f"'{user}'" for user in x[1]]))) 
# print(community_groups.take(5))

# sort the communities by size and then by user id
community_groups = community_groups.sortBy(lambda x:(x[0], x[1]))
# print(community_groups.take(5))

# collect the created communities and write to output file
formatted_communities = community_groups.map(lambda x: x[2]).collect()

with open(community_output, "w") as f:
    for community in formatted_communities:
        f.write(community + "\n")



