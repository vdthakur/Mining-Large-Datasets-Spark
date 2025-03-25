# import time
import csv
import sys
from math import sqrt
from collections import defaultdict
from pyspark import SparkContext

# start = time.time()

sctask_2 = SparkContext('local[*]', 'task2')
input_file = sys.argv[1] # "yelp_train.csv" 
validation_file = sys.argv[2] #"yelp_val.csv" 
output = sys.argv[3] # "task2_1.csv" 

yelp_rdd = sctask_2.textFile(input_file)
first_line = yelp_rdd.first()
yelp_rdd = yelp_rdd.filter(lambda line: line != first_line).map(lambda row: row.strip().split(','))


user_ratings = yelp_rdd.map(lambda row: (row[0], (float(row[2]), 1)))
# reduce by key to sum ratings and counts for each user
user_rating_totals = user_ratings.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
# calculate the mean rating for each user
user_mean_ratings = user_rating_totals.mapValues(lambda x: x[0] / x[1]).collectAsMap()

# map the users to a dictionary of businesses they rated with normalized rating (by user mean)
user_to_business_ratings = yelp_rdd.map(lambda row: (row[0], (row[1], float(row[2]) - user_mean_ratings[row[0]]))).groupByKey().mapValues(dict).collectAsMap()

# map the businesses to a dictionary of normalized (by user mean) user ratings
business_to_user_ratings = yelp_rdd.map(lambda row: (row[1], (row[0], float(row[2]) - user_mean_ratings[row[0]]))).groupByKey().mapValues(dict).collectAsMap()


# map tge (user, business) pairs to their specific actual rating
user_business_pair_ratings = yelp_rdd.map(lambda row: ((row[0], row[1]), float(row[2]))).collectAsMap()

# compute da average rating for each business
business_mean_ratings = yelp_rdd.map(lambda row: (row[1], (float(row[2]), 1))).reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1])).mapValues(lambda x: x[0] / x[1]).collectAsMap()


global_average = yelp_rdd.map(lambda row: float(row[2])).mean()

# bring in the validation file
validation_rdd = sctask_2.textFile(validation_file)
first_line = validation_rdd.first()
# map the validation rdd to have user,business
validation_rdd = validation_rdd.filter(lambda line: line != first_line).map(lambda row: row.strip().split(',')).map(lambda row: (row[0], row[1]))

# pearson correlation function
def pearson_correlation(b1, b2, business_to_user_ratings):
    # retrieve the ratings dictionaries for the two businesses
    ratings1 = business_to_user_ratings.get(b1, {})
    ratings2 = business_to_user_ratings.get(b2, {})

    # find common users who rated both businesses
    common_users = set(ratings1.keys()).intersection(ratings2.keys())
    n = len(common_users)

    if n == 0:
        return None
    # sum of the ratings for the common users for business b1
    sum1 = sum(ratings1[user] for user in common_users)
    # sum of the ratings for the common users for business b2
    sum2 = sum(ratings2[user] for user in common_users)
    # sum of squared ratings for the common users for business b1
    sum1_squared = sum(ratings1[user] ** 2 for user in common_users)
    # sum of squared ratings for the common users for business b2
    sum2_squared = sum(ratings2[user] ** 2 for user in common_users)
    # sum of the product of the ratings for the common users for business b1 and b2
    c_sum = sum(ratings1[user] * ratings2[user] for user in common_users)

    # numerator of the pearson formula
    num = c_sum - (sum1 * sum2 / n)
    # denominators of the formula
    den_p1 = sum1_squared - (sum1 ** 2) / n
    den_p2 = sum2_squared - (sum2 ** 2) / n

    # product of the variance components to form the denominator
    denominator = den_p1 * den_p2

    # checking for non posive denominator
    if denominator <= 0:
        return None

    den = sqrt(denominator)

    if den == 0:
        return None
    # pearson similarity
    similarity = num / den
    return similarity

def make_predictions(user_business_pair, user_to_business_ratings, business_to_user_ratings, user_mean_ratings, business_mean_ratings, global_average, num_neighbors):
    user, business = user_business_pair
    # in case the user or business average does not exist from the train data, we fall back on the global average of the train data
    # fall back to global averages for the user and business
    user_avg = user_mean_ratings.get(user, global_average)
    business_avg = business_mean_ratings.get(business, global_average)

    # cold start handling: both user and business are new
    if user not in user_to_business_ratings and business not in business_to_user_ratings:
        return global_average
    elif user not in user_to_business_ratings:
        # new user with only business information
        return business_avg
    elif business not in business_to_user_ratings:
        # new business with only user information
        return user_avg

    # get the ratings given by the user to other businesses
    user_ratings = user_to_business_ratings[user]
    similarities = []

    # loop through each business the user has rated to compute similarity
    for rated_business in user_ratings.keys():
        if rated_business == business:
            continue
        # compute similarity between the target business and each co-rated business
        similarity = pearson_correlation(business, rated_business, business_to_user_ratings)
        # if similarity can't be computed, approximate based on rating ratio
        if similarity is None:
            ratio = business_avg / business_mean_ratings.get(rated_business, global_average)
            # similarity based on the ratio
            if 0.9 <= ratio <= 1.1:
                similarity = 0.9
            elif 0.8 <= ratio < 0.9 or 1.1 < ratio <= 1.2:
                similarity = 0.7
            elif 0.7 <= ratio < 0.8 or 1.2 < ratio <= 1.3:
                similarity = 0.5
            elif 0.6 <= ratio < 0.7 or 1.3 < ratio <= 1.4:
                similarity = 0.3
            else:
                similarity = 0.1

        # weight negative similarities less than positive similarities
        weight = 1.0 if similarity >= 0 else 0.5 
        similarities.append((similarity * weight, user_ratings[rated_business]))
    # if similarities were not found for the user, we fallback to a weighted average of user and business averages for the prediction
    if not similarities:
        fallback = (0.6 * user_avg) + (0.4 * business_avg)
        return fallback

    # sort similarities and select top N neighbors defined by num_neighbors (found through trial and error)
    top_neighbors = sorted(similarities, key=lambda x: -abs(x[0]))[:num_neighbors]
    selected_neighbors = top_neighbors if top_neighbors else similarities

    # compute prediction using selected neighbors
    numerator = sum(similarity * rating for similarity, rating in selected_neighbors)
    denominator = sum(abs(similarity) for similarity, _ in selected_neighbors)
    # check for denominator being zero (no similarity) and fallback to user's average
    if denominator == 0:
        return user_avg
    # calculate the predicted rating and bounded it within typical rating limits
    predicted_rating = user_avg + (numerator / denominator)
    predicted_rating = max(1.0, min(5.0, predicted_rating))
    return predicted_rating

# function to make predictions for each user-business pair in the validation set
def generate_cf_predictions(user_business_pair):
    user_id, business_id = user_business_pair
    predicted_rating = make_predictions(user_business_pair,user_to_business_ratings,
        business_to_user_ratings,user_mean_ratings,
        business_mean_ratings,global_average,num_neighbors=18)
    # return the key value pair for dictionary 
    return ((user_id, business_id), predicted_rating)

# apply the function to the validation_rdd using map
cf_predictions_rdd = validation_rdd.map(generate_cf_predictions).collect()

with open(output,mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['user_id', 'business_id', 'prediction'])
    for (user_id, business_id), predicted_rating in cf_predictions_rdd:
        writer.writerow([user_id, business_id, predicted_rating])

# end = time.time()
# print(f"Execution time: {end - start} seconds")


