from data_preprocessing import data_prep
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error as mean_ae

data_movies = pd.read_csv('./ml-latest-small/movies.csv')
data_tags = pd.read_csv('./ml-latest-small/tags.csv').drop(['timestamp'], axis=1)
data_ratings = pd.read_csv('./ml-latest-small/ratings.csv').drop(['timestamp'], axis=1)

train_data, test_data, movie_train, movie_test, genere_tensor, tag_tensor, data, original_tensor, original_matrix = data_prep(data_movies, data_tags, data_ratings)

U_20, S_20, V_T_20 = np.linalg.svd(movie_test)
U_80, S_80, V_T_80 = np.linalg.svd(movie_train)
U_100, S_100, V_T_100 = np.linalg.svd(original_matrix)


k = 121

V_T_20 = V_T_20[:k, :]
V_T_80 = V_T_80[:k, :]
V_T_100 = V_T_100[:k, :]


V_20 = np.transpose(V_T_20)
V_80 = np.transpose(V_T_80)
V_100 = np.transpose(V_T_100)

concept_sim_matrix_20 = np.matmul(movie_test, V_20)
concept_sim_matrix_80 = np.matmul(movie_train, V_80)
concept_sim_matrix_100 = np.matmul(original_matrix, V_100)



concept_sim_matrix_20 = cosine_similarity(concept_sim_matrix_20)
concept_sim_matrix_80 = cosine_similarity(concept_sim_matrix_80)
concept_sim_matrix_100 = cosine_similarity(concept_sim_matrix_100)

movie_ids_20 = np.array(movie_test.index)
user_ids_20 = np.array(movie_test.columns)
movie_ids_80 = np.array(movie_train.index)
user_ids_80 = np.array(movie_train.columns)
movie_ids_100 = np.array(original_matrix.index)
user_ids_100 = np.array(original_matrix.columns)

movieId_index_20 = {}
userId_index_20 = {}
movieId_index_80 = {}
userId_index_80 = {}
movieId_index_100 = {}
userId_index_100 = {}

for i, movie_id in enumerate(movie_ids_20):
    movieId_index_20[movie_id] = i
for i, user_id in enumerate(user_ids_20):
    userId_index_20[user_id] = i

for i, movie_id in enumerate(movie_ids_80):
    movieId_index_80[movie_id] = i
for i, user_id in enumerate(user_ids_80):
    userId_index_80[user_id] = i

for i, movie_id in enumerate(movie_ids_100):
    movieId_index_100[movie_id] = i
for i, user_id in enumerate(user_ids_100):
    userId_index_100[user_id] = i

movie_train_20 = np.array(movie_test)
train_data_20 = np.array(test_data)
movie_train_80 = np.array(movie_train)
train_data_80 = np.array(train_data)
movie_train_100 = np.array(original_matrix)
train_data_100 = np.array(original_tensor)

test_data_movie_id = np.array(movie_test.index)



def get_key(value, dict):
    for key, val in dict.items():
        if val == value:
            return key

def optimized_nearest_neighbors(movieId, user_similarity_matrix, k, movieId_index):
    movie_index = movieId_index[movieId]
    user_similarity = user_similarity_matrix[movie_index]
    
    top_k_indices = np.argpartition(user_similarity, -(k+1))[-(k+1):]
    top_k_similarities = user_similarity[top_k_indices]
    
    sorted_indices = top_k_indices[np.argsort(top_k_similarities)[::-1]]
    return sorted_indices[1:k+1]

def vectorized_predict_rating(userId, movieId, movie_train, user_similarity_matrix, k, userId_index, movieId_index):
    user_index = userId_index[userId]
    movie_index = movieId_index[movieId]
    
    neighbor_indices = optimized_nearest_neighbors(movieId, user_similarity_matrix, k, movieId_index)
    
    neighbor_similarities = user_similarity_matrix[movie_index][neighbor_indices]
    neighbor_ratings = movie_train[neighbor_indices, user_index]
    
    valid_mask = neighbor_ratings != 0
    valid_similarities = neighbor_similarities[valid_mask]
    valid_ratings = neighbor_ratings[valid_mask]
    
    if len(valid_similarities) == 0:
        return 0
    
    weighted_prediction = np.sum(valid_ratings * valid_similarities) / np.sum(valid_similarities)
    return weighted_prediction

def optimized_calculate_rmse(user_ids, movie_ids, movie_train, user_similarity_matrix, k, userId_index, movieId_index):
    predictions = []
    actual = []
    
    for user_id in user_ids:
        for movie_id in movie_ids:
            if movie_train[movieId_index[movie_id], userId_index[user_id]] != 0:
                prediction = vectorized_predict_rating(user_id, movie_id, movie_train, user_similarity_matrix, k, userId_index, movieId_index)
                predictions.append(prediction)
                actual.append(movie_train[movieId_index[movie_id], userId_index[user_id]])
    
    return np.sqrt(mean_squared_error(actual, predictions)), np.mean(np.abs(np.array(actual) - np.array(predictions)))

def optimized_top_10_recommendations(user_id, movie_ids, movie_train, user_similarity_matrix, k, userId_index, movieId_index):
    recommendations = {}
    for movie_id in movie_ids:
        if movie_train[movieId_index[movie_id], userId_index[user_id]] == 0:
            prediction = vectorized_predict_rating(user_id, movie_id, movie_train, user_similarity_matrix, 200, userId_index, movieId_index)
            recommendations[movie_id] = prediction
    
    sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:10]
    return sorted_recommendations

def optimized_calculate_accuracy(user_ids, movie_ids, movie_train, test_data_movie_id, user_similarity_matrix, k, userId_index, movieId_index):

    total_precision = 0
    total_recall = 0
    total_users = 0
    
    for user_id in user_ids:
        recommendations = optimized_top_10_recommendations(user_id, movie_ids, movie_train, user_similarity_matrix, k, userId_index, movieId_index)
        recommended_movies = set(movie_id for movie_id, _ in recommendations)
        
        test_items = set(test_data_movie_id)
        
        relevant_recommended = recommended_movies.intersection(test_items)
        
        if len(recommended_movies) > 0:
            precision = len(relevant_recommended) / len(recommended_movies)
            total_precision += precision
        
        if len(test_items) > 0:
            recall = len(relevant_recommended)
            total_recall += recall
            total_users += 1
    
    avg_precision = total_precision / len(user_ids) if len(user_ids) > 0 else 0
    avg_recall = total_recall / total_users if total_users > 0 else 0
    
    f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
    
    return avg_precision, avg_recall, f1_score

def optimized_calculate_ndcg(recommendations, test_data_movie_ids):

    recommended_movies = [movie_id for movie_id, _ in recommendations][:10]
    
    relevances = [1 if movie_id in test_data_movie_ids else 0 for movie_id in recommended_movies]
    
    dcg = np.sum(np.array(relevances) / np.log2(np.arange(2, len(relevances) + 2)))
    
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = np.sum(np.array(ideal_relevances) / np.log2(np.arange(2, len(ideal_relevances) + 2)))
    
    return dcg / idcg if idcg > 0 else 0.0

def optimized_mean_ndcg(user_ids, movie_ids, movie_train, test_data_movie_id, user_similarity_matrix, k, userId_index, movieId_index):
    ndcgs = [
        optimized_calculate_ndcg(
            optimized_top_10_recommendations(user_id, movie_ids, movie_train, user_similarity_matrix, k, userId_index, movieId_index), 
            test_data_movie_id
        ) 
        for user_id in user_ids
    ]
    
    return np.mean(ndcgs)

rmse, mae = optimized_calculate_rmse(user_ids_20, movie_ids_20, movie_train_20, concept_sim_matrix_20, 200, userId_index_20, movieId_index_20)
precision, recall, f1_score = optimized_calculate_accuracy(user_ids_100, movie_ids_100, movie_train_100, test_data_movie_id, concept_sim_matrix_100, 10, userId_index_100, movieId_index_100)
ndcg = optimized_mean_ndcg(user_ids_100, movie_ids_100, movie_train_100, test_data_movie_id, concept_sim_matrix_100, 10, userId_index_100, movieId_index_100)
top_10 = optimized_top_10_recommendations(48, movie_ids_20, movie_train_20, concept_sim_matrix_20, 1900, userId_index_20, movieId_index_20)

top_10_dict = {movie_id: score for movie_id, score in top_10}
top_10_movie_ids=[movie_id[0] for movie_id in top_10]

filtered_movies = data[data['movieId'].isin([movie_id[0] for movie_id in top_10])]

filtered_movies = filtered_movies.set_index('movieId').loc[top_10_movie_ids].reset_index()

predicted_ratings=[movie_id[1] for movie_id in top_10]

filtered_movies['Predicted_rating'] = filtered_movies['movieId'].map(top_10_dict)

unique_titles = filtered_movies[['title', 'Predicted_rating']]

unique_titles = unique_titles.drop_duplicates(subset='title')

unique_titles.index = range(1, len(unique_titles) + 1)

print(f"""
Algorithm Accuracy
RMSE: {rmse:.2f}
MAE: {mae:.2f}
Precision: {precision:.2f}
Recall: {recall:.8f}
F1 Score: {f1_score:.8f}
nDCG: {ndcg:.2f}
""")
print('\n\nTop 10 Recommendations for User 1')
print(unique_titles)