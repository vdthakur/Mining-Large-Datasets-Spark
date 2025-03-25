# import time
import sys
from pyspark import SparkContext
from itertools import combinations
from collections import defaultdict, deque

# start = time.time()
sctask_2 = SparkContext('local[*]', 'task2')
filter_threshold = int(sys.argv[1])
input_f = sys.argv[2]
betweenness_out_file_path = sys.argv[3]
community_out_file_path = sys.argv[4]

# read and preprocess data
ub_data_rdd = sctask_2.textFile(input_f)
first_line = ub_data_rdd.first()

# remove header and split the data
ub_cleaned_data_rdd = ub_data_rdd.filter(lambda row:row != first_line).map(lambda row: row.strip().split(','))

# map the data to (user, business) pairs
user_b_rdd = ub_cleaned_data_rdd.map(lambda row:(row[0],row[1]))

# map the data to (business, users) pairs
business_user_rdd = user_b_rdd.map(lambda x:(x[1],x[0])).groupByKey().mapValues(set)

# create user pairs for each business who have reviewed the same business
user_pairs = business_user_rdd.flatMap(lambda x: [(tuple(sorted((u1, u2))), 1)for u1, u2 in combinations(x[1], 2)])
# print(user_pairs.take(5))

# count the number of times each user pair appears
user_pair_counts = user_pairs.reduceByKey(lambda x,y: x + y)
# print(user_pair_counts.take(5))

# filter pairs that appear more than filter_threshold, defined as system argument
edges_rdd = user_pair_counts.filter(lambda x: x[1] >= filter_threshold).map(lambda x: x[0])
# create undirected edges (a,b) and (b,a) for each edge
edges_rdd = edges_rdd.flatMap(lambda x:[x,(x[1],x[0])])
# print(edges_rdd.take(5))

# create adjaceny rdd by grouping users with those who have been filtered by a threshold to review the same business as them ie : user3: {user1,user2}
adjacency_list_rdd = edges_rdd.groupByKey().mapValues(set)
# print(adjacency_list_rdd.take(5))   
adjacency_list = adjacency_list_rdd.collectAsMap()
adjacency_list_broadcasted = sctask_2.broadcast(adjacency_list)

# create the nodes (vertices) rdd using the unique users from the adjacency list which are the nodes post filtering
nodes = list(adjacency_list.keys())
nodes_rdd = sctask_2.parallelize(nodes)

def calc_betweenness(start_node,graph):
    # definne a list to store the order of visits
    visit_order = []  
    # define a dictionary to store the nodes on shortest path to each node
    shortest_ck_points = defaultdict(list)
    # define a dictionary to store the number of shortest paths to each node

    path_count = defaultdict(int)  
    # define a dictionary to store the shortest path distance to each node
    shortest_path_distance = {}

    # define the path count of the start node as 1
    path_count[start_node] = 1  
    # define the shortest path distance of the start node as 0 to begin with
    shortest_path_distance[start_node] = 0
    # start a deque with the start node
    search_queue = deque([start_node])

    # while there are nodes in the queue
    while search_queue:
        # get the next node to process
        current_node = search_queue.popleft()
        # append this node to visit order to note it has been visited
        visit_order.append(current_node)
        # for each neighbor of the current node
        for neighbor in graph.get(current_node,set()):
            # if the neighbor has not been visited
            if neighbor not in shortest_path_distance:
                # add the neighbor to the search queue
                search_queue.append(neighbor)
                # set the shortest path distance of the neighbor to the current node + 1
                shortest_path_distance[neighbor]= shortest_path_distance[current_node] + 1
            # if the shortest path distance of the neighbor is the same as the current node + 1
            if shortest_path_distance[neighbor] == shortest_path_distance[current_node]+ 1:
                # add the path count of the neighbor to the current node
                path_count[neighbor] += path_count[current_node]
                # add the current node to the shortest check points of the neighbor
                shortest_ck_points[neighbor].append(current_node)

    # compute dependencies contribtued by each node
    node_dependency = defaultdict(float)  

    while visit_order:
        # get a node from the visited order in reverse
        node = visit_order.pop()  
        # for each check point node in the shortest path of node
        for on_path_point in shortest_ck_points[node]:

            # calculate the dependency of the previous check points (how much dependency the current node contributes back to each previous node in shortest_ck_points)
            # dependency is proportional to the number of shortest paths from the start node that go through this previous point to reach the current node
            p_dependency = (path_count[on_path_point] / path_count[node]) * (1 + node_dependency[node])

            # create a sorted tuple representing the edge
            edge= tuple(sorted((on_path_point,node))) 

            # accumulate the dependency for the on_path_point
            node_dependency[on_path_point] += p_dependency  

            #  yield the edge and its contribution to the node
            yield (edge,p_dependency)  

def create_communities(adjacency_list):

    # set to track visited nodes
    visited_nodes = set() 

    # list to store each community as a set of nodes
    communities = []  

    # for each node in the adjacency list
    for start_node in adjacency_list.keys():

        # if the node hasn't been visited, initiate a search for community
        if start_node not in visited_nodes:

            # set created to store nodes in the current community
            current_community = set() 

            # initiate the deque with the start node
            node_queue = deque([start_node])  

            # note the start node as visited
            visited_nodes.add(start_node) 

            # while there are nodes in the queue
            while node_queue:
                # get the next node to process
                current_node = node_queue.popleft() 
                # add the node to the current community
                current_community.add(current_node) 

                # for each neighor of the current
                for neighbor in adjacency_list.get(current_node, set()):
                    # if the neighbor hasn't been visited
                    if neighbor not in visited_nodes:  
                        # now mark the neighbor as visited
                        visited_nodes.add(neighbor)
                        # add it to the queue
                        node_queue.append(neighbor)  
            # append the created community to the list
            communities.append(current_community)  

    # if there are any nodes that werent visited they are isolated
    single_nodes = set(adjacency_list.keys()) - visited_nodes 

    # for each single node left over we form a community
    for sin_node in single_nodes:
        communities.append({sin_node})  

    # return the communities list
    return communities 

# the number of edges found in the original graph
n_edges = edges_rdd.count() // 2

# create a dictionary of degrees for each node
degrees = {node:len(neighbors) for node,neighbors in adjacency_list.items()}

# create a copy of the adjacency list
current_adjacency = {node:set(neighbors) for node,neighbors in adjacency_list.items()}

# set the best communities and modularity score to empty and negative infinity for comparison and reevaluation
best_coms = []
best_mod = float('-inf')

# round = 0

while True:
    # round += 1
    # print(f"Round {round}")

    # compute the betweenness of each edge
    betweenness = nodes_rdd.flatMap(lambda node: calc_betweenness(node,current_adjacency))
    betweenness = betweenness.reduceByKey(lambda x,y: x + y).mapValues(lambda x: x /2.0)

    # collect the betweenness values
    edge_betweenness = betweenness.collect()

    # removing until no edges remain so loop until edge_betweenness is not empty
    if not edge_betweenness:
        break  

    # find the edge which has the maximum betweenness
    max_betweenness = max(edge_betweenness,key=lambda x: x[1])[1]

    # remove edges with the highest betweenness
    edges_to_remove = [edge for edge,value in edge_betweenness if value == max_betweenness]

    # for each edge to remove
    for edge in edges_to_remove:
        # define the nodes of the edge
        node1, node2 = edge
        # if node2 in the adjacency list
        if node2 in current_adjacency.get(node1,set()):
            # remove the edge from the adjacency list
            current_adjacency[node1].remove(node2)
            # if the node has no neighbors, remove it from the adjacency list
            if not current_adjacency[node1]:
                del current_adjacency[node1]
        # if node1 in the adjacency list
        if node1 in current_adjacency.get(node2,set()):
            current_adjacency[node2].remove(node1)
            if not current_adjacency[node2]:
                del current_adjacency[node2]

    # call the create_communities function to find the communities in the current graph
    communities = create_communities(current_adjacency)

    def find_modularity(communities, current_adjacency_dict,total_edges,original_degrees):
    # set the original modularity score to zero
        modularity_score =0.0  
        # iterate over the communities found
        for community in communities:
            # for each node in the community
            for node_i in community:
                # retrieve the node's degree in the original graph
                degree_node_i = original_degrees[node_i]  
                # for all nodes in the community
                for node_j in community:
                    # find the degree of the node from the original graph
                    degree_node_j = original_degrees[node_j]  
                    # hint#2: A is 1 if both node_i is connected to node_j and node_j to node_i in current graph
                    edge_present = 1 if node_j in current_adjacency_dict.get(node_i,set()) and node_i in current_adjacency_dict.get(node_j,set()) else 0
                    # using the modularity equation caclulate the modularity score (Aij - (ki*kj)/2m)
                    modularity_score += edge_present - (degree_node_i * degree_node_j) /(2 * total_edges)

        # divide the modularity score by 2m
        modularity_score /= (2*total_edges)

        return modularity_score
    
    # calculate the modularity of the current communities
    modularity = find_modularity(communities, current_adjacency, n_edges,degrees)
    # print(f"Modularity {modularity}")

    # update the best communities and modularity if the current modularity is greater
    if modularity > best_mod :
        best_coms = communities
        best_mod= modularity

    # if nothing more exists in the adjacency list, break the loop
    if not current_adjacency:
        break

# calculate betweeness for the original graph
betweenness = nodes_rdd.flatMap(lambda node: calc_betweenness(node,adjacency_list_broadcasted.value))
betweenness = betweenness.reduceByKey(lambda x,y: x + y).mapValues(lambda x: x /2.0)
edge_betweenness = betweenness.collect()
edge_betweenness_sorted = sorted(edge_betweenness,key=lambda x:(-x[1], x[0]))

# write to file
with open(betweenness_out_file_path, "w") as f:
    for edge, betweenness in edge_betweenness_sorted:
        f.write(f"('{edge[0]}', '{edge[1]}'),{round(betweenness, 5)}\n")

sorted_communities = [sorted(list(community)) for community in best_coms]
sorted_communities.sort(key=lambda x: (len(x), x))
# write the communities to the output file
with open(community_out_file_path, "w") as f:
    for community in sorted_communities:
        formatted_community = ', '.join([f"'{user}'" for user in community])
        f.write(f"{formatted_community}\n")

# end = time.time()
# print(f"Time: {end - start:.2f}")


