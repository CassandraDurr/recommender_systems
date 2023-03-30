"""Functions required for prac 2, recommender system with explicit rating data."""
import gc
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
from functions import reg_logll, rmse, index_data, create_ratings_df

gc.collect()

# Combine dataframes on "movieId"
ratings = create_ratings_df(file_name = "ratings_small.csv")

user_idx, user_start_index, user_end_index = index_data(
    ratings, "rating_10", "userId", "movieId_order"
)
# Drop the rating column
user_idx = user_idx[:,0:2]

# Movie frequencies
# Obtain frequencies per id
movie_frequencies = (
    ratings["movieId_order"]
    .value_counts(normalize=True)
    .reset_index()
    .sort_values(by="index")
    .reset_index(drop=True)
)
movie_frequencies.columns = ["movieId_order", "frequency"]
# movie_frequencies = np.array(movie_frequencies.frequency)
# print(dict(zip(movie_frequencies.movieId_order, movie_frequencies.frequency)))
print(np.array(movie_frequencies.frequency))

# Initialise U and V and bias vectors
latentDim = 12
# M=162541, N=59047
M = ratings["userId"].nunique()
N = ratings["movieId"].nunique()

# Delete ratings from memory to save some space now that all operations have been done with it
del ratings
gc.collect()

lmd = 0.1
tau = 0.01
alpha = 0.01
# Randomly initialise U, V, b_m and b_n
U = np.random.normal(scale=5 / np.sqrt(latentDim), size=latentDim * M).reshape(
    M, latentDim
)
V = np.random.normal(scale=5 / np.sqrt(latentDim), size=latentDim * N).reshape(
    N, latentDim
)
b_m = np.zeros(M)
b_n = np.zeros(N)

user_subset = user_idx[user_start_index[0] : user_end_index[0]]
# print(user_subset)
# Acceptable negtives
neg_options = set(np.arange(N)).difference(set(user_subset[:,1]))
# print(neg_options)