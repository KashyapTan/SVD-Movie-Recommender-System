from data_preprocessing import data_prep
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
import array
from sklearn.metrics import mean_squared_error, mean_absolute_error as mae


data_movies = pd.read_csv('./ml-latest-small/movies.csv')
data_tags = pd.read_csv('./ml-latest-small/tags.csv').drop(['timestamp'], axis=1)
data_ratings = pd.read_csv('./ml-latest-small/ratings.csv').drop(['timestamp'], axis=1)


train_data, test_data, movie_train, movie_test, genere_tensor, tag_tensor, data = data_prep(data_movies, data_tags, data_ratings)

movie_ids = np.array(movie_train.index)
user_ids = np.array(movie_train.columns)
movieId_index = {}
userId_index = {}

for i, movie_id in enumerate(movie_ids):
    movieId_index[movie_id] = i
for i, user_id in enumerate(user_ids):
    userId_index[user_id] = i

movie_train = np.array(movie_train)
train_data = np.array(train_data)

user_sim_matrix = cosine_similarity(train_data)

def nearest_neighbors(movieId, user_similarity_matrix, k):
    movie_index = movieId_index[movieId]
    answer = {}
    result = []
    user_similarity = user_similarity_matrix[movie_index]
    neighbors = sorted(user_similarity, reverse=True)
    for i in range(k):
        result.append(neighbors[i+1])
        answer[neighbors[i+1]] = -1

    for i in range(len(user_similarity)):
        if user_similarity[i] in result:
            answer[user_similarity[i]] = i
    return answer

def get_key(value, dict):
    for key, val in dict.items():
        if val == value:
            return key



# FIX MOVIE ID IT SHOULD BE BY INDEX NOT BY ID BECAUSE 2D ARRAY
def predict_rating(userId, movieId, movie_train, user_simalarity_matrix, k):
    user_index = userId_index[userId]
    movieId_similarity = nearest_neighbors(movieId, user_simalarity_matrix, k) 
    neighbors = movieId_similarity.values()

    for neighbor in neighbors:
        movieId_similarity[get_key(neighbor, movieId_similarity)] = movie_train[neighbor][user_index]

    ratings = list(movieId_similarity.values())
    sim_values = list(movieId_similarity.keys())

    predection = 0
    sum_sim_values = 0

    # Ignoring zeroes because there are minimal ratings for certain movies
    for i in range(len(ratings)):
        if ratings[i] != 0:
            predection += ratings[i] * sim_values[i]
            sum_sim_values += sim_values[i]
    if sum_sim_values == 0:
        return 0, userId, movieId        
    predection /= sum_sim_values
    return predection, userId, movieId


predection, user_id, movie_id = predict_rating(userId=4, movieId=296, movie_train=movie_train, user_simalarity_matrix=user_sim_matrix, k=10)
print(f"UU | User {user_id} will rate movie {movie_id} a {predection:.2f}/5")
print(f"User {user_id} actual rating of {movie_id} is {movie_train[movieId_index[movie_id]][userId_index[user_id]]}/5")

# Calculate the RMSE for the model

def calculate_rmse(user_ids, movie_ids, movie_train, user_similarity_matrix, k):
    predictions = []
    actual = []
    for user_id in user_ids:
        for movie_id in movie_ids:
            if movie_train[movieId_index[movie_id]][userId_index[user_id]] != 0:
                predection, _, _ = predict_rating(userId=user_id, movieId=movie_id, movie_train=movie_train, user_simalarity_matrix=user_similarity_matrix, k=k)
                predictions.append(predection)
                actual.append(movie_train[movieId_index[movie_id]][userId_index[user_id]])
    return np.sqrt(mean_squared_error(actual, predictions)), mae(actual, predictions)

rmse, mae = calculate_rmse(user_ids, movie_ids, movie_train, user_sim_matrix, 10)

print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")


