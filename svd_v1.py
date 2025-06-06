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


def nearest_neighbors(movieId, user_similarity_matrix, k, movieId_index):
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



#* FIX MOVIE ID IT SHOULD BE BY INDEX NOT BY ID BECAUSE 2D ARRAY
def predict_rating(userId, movieId, movie_train, user_simalarity_matrix, k, userId_index, movieId_index):
    user_index = userId_index[userId]
    movieId_similarity = nearest_neighbors(movieId, user_simalarity_matrix, k, movieId_index) 
    neighbors = movieId_similarity.values()

    for neighbor in neighbors:
        movieId_similarity[get_key(neighbor, movieId_similarity)] = movie_train[neighbor][user_index]

    ratings = list(movieId_similarity.values())
    sim_values = list(movieId_similarity.keys())

    predection = 0
    sum_sim_values = 0

    #* Ignoring zeroes because there are minimal ratings for certain movies
    for i in range(len(ratings)):
        if ratings[i] != 0:
            predection += ratings[i] * sim_values[i]
            sum_sim_values += sim_values[i]
    if sum_sim_values == 0:
        return 0, userId, movieId        
    predection /= sum_sim_values
    return predection, userId, movieId



def calculate_rmse(user_ids, movie_ids, movie_train, user_similarity_matrix, k, userId_index, movieId_index):
    predictions = []
    actual = []
    for user_id in user_ids:
        for movie_id in movie_ids:
            if movie_train[movieId_index[movie_id]][userId_index[user_id]] != 0:
                predection, _, _ = predict_rating(user_id, movie_id, movie_train, user_similarity_matrix, k, userId_index, movieId_index)
                predictions.append(predection)
                actual.append(movie_train[movieId_index[movie_id]][userId_index[user_id]])
    return np.sqrt(mean_squared_error(actual, predictions)), mean_ae(actual, predictions)

def top_10_recommmendations(user_id, movie_ids, movie_train, user_similarity_matrix, k, userId_index, movieId_index):
    rec={}
    for movie_id in movie_ids:
        if movie_train[movieId_index[movie_id]][userId_index[user_id]] == 0:
            predection, _, _ = predict_rating(user_id, movie_id, movie_train, user_similarity_matrix, 200, userId_index, movieId_index)
            rec[movie_id]=predection
    top_10 = sorted(rec.items(), key=lambda item: item[1], reverse=True)[:k]
    return top_10

def calculate_accuracy(user_ids, movie_ids, movie_train, test_data_movie_id, user_similarity_matrix, k, userId_index, movieId_index):
    precisions = []
    recalls = []
    
    
    for user_id in user_ids:
        test_movies = set([movie_id for movie_id in movie_ids if movie_train[movieId_index[movie_id]][userId_index[user_id]] != 0])
        recommendations = top_10_recommmendations(user_id, movie_ids, movie_train, user_similarity_matrix, k, userId_index, movieId_index)
        recommended_movies = set([movie_id for movie_id, _ in recommendations])
        
        overlap = len(recommended_movies.intersection(test_movies))
        
        if overlap > 0:
            precision = overlap / len(recommended_movies)
            recall = overlap / len(test_movies)
            
            precisions.append(precision)
            recalls.append(recall)
    
    mean_precision = np.mean(precisions) if precisions else 0
    mean_recall = np.mean(recalls) if recalls else 0
    
    if mean_precision + mean_recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (mean_precision * mean_recall) / (mean_precision + mean_recall)
        
    return mean_precision, mean_recall, f1_score

def calculate_ndcg(recommendations, test_data_movie_ids):
    recommended_movies = [movie_id for movie_id, _ in recommendations][:10]
    
    relevances = [1 if movie_id in test_data_movie_ids else 0 for movie_id in recommended_movies]
    
    dcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevances))
    
    ideal_relevances = sorted(relevances, reverse=True)
    
    idcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_relevances))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def calculate_mean_ndcg(user_ids, movie_ids, movie_train, test_data_movie_id, user_similarity_matrix, k, userId_index, movieId_index):
    ndcgs = []
    for user_id in user_ids:
        recommendations = top_10_recommmendations(user_id, movie_ids, movie_train, user_similarity_matrix, k, userId_index, movieId_index)
        ndcg = calculate_ndcg(recommendations, test_data_movie_id)
        ndcgs.append(ndcg)
    
    return np.mean(ndcgs) if ndcgs else 0.0


rmse, mae = calculate_rmse(user_ids_20, movie_ids_20, movie_train_20, concept_sim_matrix_20, 200, userId_index_20, movieId_index_20)
precision, recall, f1_score= calculate_accuracy(user_ids_100, movie_ids_100, movie_train_100, test_data_movie_id, concept_sim_matrix_100, 10, userId_index_100, movieId_index_100)
ndcg = calculate_mean_ndcg(user_ids_100, movie_ids_100, movie_train_100, test_data_movie_id, concept_sim_matrix_100, 10, userId_index_100, movieId_index_100)
top_10 = top_10_recommmendations(48, movie_ids_20, movie_train_20, concept_sim_matrix_20, 1900, userId_index_20, movieId_index_20)

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

