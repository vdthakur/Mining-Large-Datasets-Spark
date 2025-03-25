# import time
import sys
import csv
from math import sqrt
from collections import defaultdict
from pyspark import SparkContext
import numpy as np
import xgboost as xgb
import json
from datetime import datetime
# from sklearn.model_selection import GridSearchCV

# start = time.time()
sctask_2_2 = SparkContext('local[*]', 'task2')

# file paths
folder_path = sys.argv[1] # "../resource/asnlib/publicdata/" 
test_set = sys.argv[2] # "yelp_val.csv"
output = sys.argv[3] # "task2_2.csv"
input_file = folder_path + "/yelp_train.csv"
user = folder_path + "/user.json"
business = folder_path + "/business.json"

# load data to rdd
yelp_rdd = sctask_2_2.textFile(input_file)
first_line = yelp_rdd.first()  
yelp_rdd = yelp_rdd.filter(lambda line: line != first_line).map(lambda row: row.strip().split(','))
yelp_rdd = yelp_rdd.map(lambda x: (x[0], x[1], float(x[2])))  

# load user and business data with data of interest (ie. user_id, review_count, average_stars, yelping_since, business_id, review_count, stars)
user_rdd = sctask_2_2.textFile(user)
user_rdd = user_rdd.map(lambda x: json.loads(x)).map(lambda x: (x['user_id'],(x.get('review_count', 0), x.get('average_stars', None), x.get('yelping_since', None)))).collectAsMap()
business_rdd = sctask_2_2.textFile(business)
business_rdd = business_rdd.map(lambda x: json.loads(x)).map(lambda x: (x['business_id'],(x.get('review_count', 0), x.get('stars', None)))).collectAsMap()

# function to calculate time on yelp
def calculate_time_since(last_review_date, current_date=datetime.now()):
    if last_review_date:
        last_date = datetime.strptime(last_review_date, '%Y-%m-%d')  
        return (current_date - last_date).days  
    else:
        return None  

# create train set with the files and features defined above
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
    # features are user avg, user count, business avg, business count, time been on yelp)
    X_train = [user, business, user_avg, user_count, business_avg, business_count, time_yelping]
    y_train = rating
    return X_train, y_train

# initiate the test set rdd
test_rdd = sctask_2_2.textFile(test_set)
f_line = test_rdd.first()  
test_rdd = test_rdd.filter(lambda line: line != f_line).map(lambda row: row.strip().split(','))

test_rdd = test_rdd.map(lambda x: (x[0], x[1])) 

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

# create the training and test sets
train_set = yelp_rdd.map(create_train_set).collect()
test_sets = test_rdd.map(create_test_set).collect()

# separate features and labels
X_train = np.array([x[0][2:] for x in train_set])  
y_train = np.array([x[1] for x in train_set])

X_test = np.array([x[0][2:] for x in test_sets])  
y_test = np.array([x[1] for x in test_sets])


# train the model using hyperparameters found from gridsearch
xgb_model = xgb.XGBRegressor(objective='reg:linear', n_estimators=200, max_depth=5, gamma=0.2, learning_rate=0.1, colsample_bytree=0.6, reg_alpha=0.1, reg_lambda=1.5, subsample=1.0)

xgb_model.fit(X_train, y_train)

# predict on test set
predictions = xgb_model.predict(X_test)

# combine the predictions with the user and business ids
X_test_labels = np.array([x[0][:2] for x in test_sets])  
predicts = np.c_[X_test_labels, predictions]

# write the predictions to a csv file
with open(output,mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['user_id', 'business_id', 'prediction'])
    for prediction in predicts:
        writer.writerow([prediction[0], prediction[1], prediction[2]])


# end = time.time()
# print(f"Time taken: {end - start} seconds")



# gridsearch code to find best parameters for the model
# xgb_model = xgb.XGBRegressor(objective='reg:squarederror')
# param_grid = {'n_estimators': [50, 100, 200],'max_depth': [3, 5, 7],'learning_rate': [0.01, 0.1, 0.2],'subsample': [0.6, 0.8, 1.0],'colsample_bytree': [0.6, 0.8, 1.0],'gamma': [0, 0.1, 0.2],
#     'reg_alpha': [0, 0.01, 0.1],'reg_lambda': [1, 1.5, 2]}

# grid_search = GridSearchCV(
#     estimator=xgb_model,
#     param_grid=param_grid,
#     scoring='neg_mean_squared_error', cv=3,verbose=2,n_jobs=-1)

# grid_search.fit(X_train, y_train)
# best_model = grid_search.best_estimator_

# print("Best parameters found:")
# for param, value in grid_search.best_params_.items():
#     print(f"{param}: {value}")

# print("Best RMSE (negative): ", grid_search.best_score_)


