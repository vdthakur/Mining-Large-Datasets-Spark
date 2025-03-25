# import time
import sys
import csv
from math import sqrt
from pyspark import SparkContext
import numpy as np
import xgboost as xgb
import json
from datetime import datetime

# start = time.time()

sctask_2_3 = SparkContext('local[*]', 'task3')
folder_path = sys.argv[1] # "../resource/asnlib/publicdata/" 
test_set = sys.argv[2] # "yelp_val.csv"
output = sys.argv[3] # "task2_2.csv"
input_file = folder_path + "/yelp_train.csv"
user = folder_path + "/user.json"
business = folder_path + "/business.json"

# load data to rdd
yelp_rdd = sctask_2_3.textFile(input_file)
first_line = yelp_rdd.first()
yelp_rdd = yelp_rdd.filter(lambda line: line != first_line).map(lambda row: row.strip().split(','))
yelp_rdd = yelp_rdd.map(lambda x: (x[0], x[1], float(x[2])))

# mean rating for each user
user_ratings = yelp_rdd.map(lambda row: (row[0], (float(row[2]), 1)))
# reduce by key to sum ratings and counts for each user
user_rating_totals = user_ratings.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
# calculate the mean rating for each user
user_mean_ratings = user_rating_totals.mapValues(lambda x: x[0] / x[1]).collectAsMap()

# map da users to a dictionary of businesses they rated
user_to_business_ratings = yelp_rdd.map(lambda row: (row[0], (row[1], float(row[2]) - user_mean_ratings[row[0]]))).groupByKey().mapValues(dict).collectAsMap()

# map da businesses to a dictionary of normalized user ratings
business_to_user_ratings = yelp_rdd.map(lambda row: (row[1], (row[0], float(row[2]) - user_mean_ratings[row[0]]))).groupByKey().mapValues(dict).collectAsMap()

# map da (user, business) pairs to their specific rating
user_business_pair_ratings = yelp_rdd.map(lambda row: ((row[0], row[1]), float(row[2]))).collectAsMap()

# compute the average rating for each business
business_avg_ratings = yelp_rdd.map(lambda row: (row[1], (float(row[2]), 1))).reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1])).mapValues(lambda x: x[0] / x[1]).collectAsMap()

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

global_average = yelp_rdd.map(lambda row: float(row[2])).mean()

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

    # Get the ratings given by the user to other businesses
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

validation_rdd = sctask_2_3.textFile(test_set)
first_line = validation_rdd.first()
validation_rdd = validation_rdd.filter(lambda line: line != first_line).map(lambda row: row.strip().split(',')).map(lambda row: (row[0], row[1]))


def generate_cf_predictions(user_business_pair):
    user_id, business_id = user_business_pair
    predicted_rating = make_predictions(
        user_business_pair,
        user_to_business_ratings,
        business_to_user_ratings,
        user_mean_ratings,
        business_avg_ratings,
        global_average,num_neighbors=15)
    # return the key-value pair for dictionary
    return ((user_id, business_id), predicted_rating)

# apply the function to the validation_rdd 
cf_predictions_rdd = validation_rdd.map(generate_cf_predictions)
cf_predictions = cf_predictions_rdd.collectAsMap()

# Load user and business data, including last review dates
user_rdd = sctask_2_3.textFile(user)
user_rdd = user_rdd.map(lambda x: json.loads(x)).map(lambda x: (x['user_id'],(x.get('review_count', 0), x.get('average_stars', None), x.get('yelping_since', None)))).collectAsMap()  
business_rdd = sctask_2_3.textFile(business)
business_rdd = business_rdd.map(lambda x: json.loads(x)).map(lambda x: (x['business_id'],(x.get('review_count', 0), x.get('stars', None)))).collectAsMap()


def calculate_time_since(last_review_date, current_date=datetime.now()):
    if last_review_date:
        last_date = datetime.strptime(last_review_date, '%Y-%m-%d')  
        return (current_date - last_date).days  # time difference in days
    else:
        return None

def create_train_set(user_business_rating):
    user = user_business_rating[0]
    business = user_business_rating[1]
    rating = user_business_rating[2]

    if user in user_rdd and business in business_rdd:
        user_avg = user_rdd[user][1]
        user_count = user_rdd[user][0]
        user_yelping_since = user_rdd[user][2]
        business_avg = business_rdd[business][1]
        business_count = business_rdd[business][0]
        time_yelping = calculate_time_since(user_yelping_since)
    elif user in user_rdd:
        user_avg = user_rdd[user][1]
        user_count = user_rdd[user][0]
        user_yelping_since = user_rdd[user][2]
        business_avg = 3
        business_count = None
        time_yelping = calculate_time_since(user_yelping_since)
    elif business in business_rdd:
        user_avg = 3
        user_count = None
        user_yelping_since = None
        business_avg = business_rdd[business][1]
        business_count = business_rdd[business][0]
        time_yelping = None
    else:
        user_avg, user_count, business_avg, business_count = None, None, None, None
        time_yelping = None

    X_train = [user, business, user_avg, user_count, business_avg, business_count, time_yelping]
    y_train = rating
    return X_train, y_train



# create test set in the same way as the train set
def create_test_set(user_business):
    user = user_business[0]
    business = user_business[1]

    if user in user_rdd and business in business_rdd:
        user_avg = user_rdd[user][1]
        user_count = user_rdd[user][0]
        user_yelping_since = user_rdd[user][2]
        business_avg = business_rdd[business][1]
        business_count = business_rdd[business][0]
        time_yelping = calculate_time_since(user_yelping_since)
    elif user in user_rdd:
        user_avg = user_rdd[user][1]
        user_count = user_rdd[user][0]
        user_yelping_since = user_rdd[user][2]
        business_avg = 3
        business_count = None
        time_yelping = calculate_time_since(user_yelping_since)
    elif business in business_rdd:
        user_avg = 3
        user_count = None
        user_yelping_since = None
        business_avg = business_rdd[business][1]
        business_count = business_rdd[business][0]
        time_yelping = None
    else:
        user_avg, user_count, business_avg, business_count = None, None, None, None
        time_yelping = None

    X_test = [user, business, user_avg, user_count, business_avg, business_count, time_yelping]
    y_test = 0
    return X_test, y_test

# prepare training and test sets
train_set = yelp_rdd.map(create_train_set).collect()
test_sets = validation_rdd.map(create_test_set).collect()

# Extract features and labels
X_train = np.array([x[0][2:] for x in train_set])  
y_train = np.array([x[1] for x in train_set])

X_test = np.array([x[0][2:] for x in test_sets])  
y_test = np.array([x[1] for x in test_sets])

# train model using hyperparameters found from gridsearch
xgb_model = xgb.XGBRegressor(objective='reg:linear', n_estimators=200, max_depth=5, gamma=0.2, learning_rate=0.1, colsample_bytree=0.6, reg_alpha=0.1, reg_lambda=1.5, subsample=1.0)

xgb_model.fit(X_train, y_train)

# compile predictions into dictionary
model_predictions = {(test_sets[i][0][0], test_sets[i][0][1]): pred for i, pred in enumerate(xgb_model.predict(X_test, output_margin=True))}

# define hybrid system weights
weight_cf = 0.06 
weight_model = 1 - weight_cf

# missing_predictions = []

def create_final_prediction(user_business_pair):
    user_id, business_id = user_business_pair
    key = (user_id, business_id)
    pred_cf = cf_predictions.get(key)
    pred_model = model_predictions.get(key)

    if pred_cf is None and pred_model is None:
        final_pred = global_average 
        # missing_predictions.append(key)  
    elif pred_cf is None:
        final_pred = pred_model
    elif pred_model is None:
        final_pred = pred_cf
    else:
        final_pred = weight_cf * pred_cf + weight_model * pred_model

    return (user_id,business_id,final_pred)

# transform validation_rdd to create final predictions
final_predictions = validation_rdd.map(create_final_prediction).collect()

# if missing_predictions:
#     print(f"Predictions were missing for {len(missing_predictions)} user-business pairs.")


with open(output,mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['user_id', 'business_id', 'prediction'])
    for user_id, business_id, final_pred in final_predictions:
        writer.writerow([user_id, business_id, final_pred])

# end = time.time()

# print("Duration: ", end - start)










