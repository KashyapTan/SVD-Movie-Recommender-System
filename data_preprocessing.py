import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split


data_movies = pd.read_csv('./ml-latest-small/movies.csv')
data_tags = pd.read_csv('./ml-latest-small/tags.csv').drop(['timestamp'], axis=1)
data_ratings = pd.read_csv('./ml-latest-small/ratings.csv').drop(['timestamp'], axis=1)

def data_prep(data_movies, data_tags, data_ratings):
    data = pd.merge(data_movies, data_ratings, on=['movieId'],)
    data = pd.merge(data, data_tags, on=['userId', 'movieId'], how='left')

    movie_matrix = data.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
    data_movies['genres'] = data_movies['genres'].str.split('|')

    generes = set([genere for generes in data_movies['genres'] for genere in generes])

    genere_idx = {genere: i for i, genere in enumerate(generes)}


    data_tags = data_tags['tag'].astype('category')
    tag_index = {tag: i for i, tag in enumerate(data_tags.cat.categories)}

    num_movies = len(data_movies)
    num_generes = len(genere_idx)
    num_tags = len(tag_index)

    genere_tensor = torch.zeros((num_movies, num_generes), dtype=torch.float32)

    for i, genere in enumerate(data_movies['genres']):
        for gen in genere:
            genere_tensor[i, genere_idx[gen]] = 1.0

    tag_tensor = torch.zeros((len(data_tags), num_tags), dtype=torch.float32)

    for i, tag in enumerate(data_tags):
        tag_tensor[i, tag_index[tag]] = 1.0


    def normalize_ratings(movie_matrix):
        movie_matrix.apply(lambda row: row.apply(lambda x: x - (row.sum() / (row != 0).sum()) if x != 0 else x), axis=1)
        return movie_matrix
    normalized_matrix = normalize_ratings(movie_matrix)
    movie_tensor = torch.tensor(normalized_matrix.values, dtype=torch.float32)

    original_tensor = movie_tensor.clone()
    original_matrix = movie_matrix.copy()

    train_data, test_data = train_test_split(movie_tensor, test_size=0.2, random_state=30)
    movie_train, movie_test = train_test_split(movie_matrix, test_size=0.2, random_state=30)

    movie_train = np.transpose(movie_train)
    train_data = np.transpose(train_data)
    movie_test = np.transpose(movie_test)
    test_data = np.transpose(test_data)

    return train_data, test_data, movie_train, movie_test, genere_tensor, tag_tensor, data, original_tensor, original_matrix


