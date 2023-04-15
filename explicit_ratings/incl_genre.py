"""Explicit ratings model that includes genre information."""
import gc
import pandas as pd
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from functions import (
    map_genres,
    create_ratings_df,
    index_data,
    rmse,
    reg_logll_with_genre,
)

# Read in genre information
# columns: movieId,movieId_order,title,genres,genres_v2
# genres_v2 is contains lists of strings for the sets of movie genres
movie_genres = pd.read_csv(
    "movies_25m_genres_full.csv", converters={"genres_v2": pd.eval}
)
# Drop irrelevant columns and rename 'genres_v2
movie_genres.drop(columns=["Unnamed: 0", "genres", "movieId", "title"], inplace=True)
movie_genres.rename(columns={"genres_v2": "genre_names"}, inplace=True)

# Mapping from genre name to integers
genres = {
    "Horror": 0,
    "War": 1,
    "Crime": 2,
    "IMAX": 3,
    "Western": 4,
    "Children": 5,
    "Adventure": 6,
    "(no genres listed)": 7,
    "Drama": 8,
    "Romance": 9,
    "Thriller": 10,
    "Film-Noir": 11,
    "Animation": 12,
    "Comedy": 13,
    "Sci-Fi": 14,
    "Fantasy": 15,
    "Action": 16,
    "Musical": 17,
    "Mystery": 18,
    "Documentary": 19,
}

movie_genres["genre_values"] = movie_genres["genre_names"].apply(
    lambda row: map_genres(row, genres)
)
movie_genres["genre_count"] = movie_genres["genre_values"].apply(len)
movie_genres.drop(columns=["genre_names"], inplace=True)
# Now we only have columns: movieId_order, genre_values, genre_count

# Create a dictionary where movie_Id_order is the key
movie_dict = movie_genres[["genre_values", "genre_count"]].to_dict(orient="index")

# For the update equations it is also useful to have the genre_values as keys
# and the movie ids as items
genre_dict = {}
for index, row in movie_genres.iterrows():
    # for each genre in a movies list of genres
    for genre_value in row["genre_values"]:
        # if the key already exists
        if genre_value in genre_dict:
            genre_dict[genre_value].append(row["movieId_order"])
        # if the key does not exist
        else:
            genre_dict[genre_value] = [row["movieId_order"]]

# Create ratings dataframe, as before
ratings = create_ratings_df(file_name="ratings_25m.csv")

# Create indices
user_ratings, user_start_index, user_end_index = index_data(
    ratings, "rating_10", "userId", "movieId_order"
)
movie_ratings, movie_start_index, movie_end_index = index_data(
    ratings, "rating_10", "movieId_order", "userId"
)

# Initialise U, V, F and bias vectors
latentDim = 12
M = ratings["userId"].nunique()
N = ratings["movieId"].nunique()

# Delete ratings from memory to save some space now that all operations have been done with it
del ratings
gc.collect()

lmd = 0.1
tau = 0.01
alpha = 0.01
beta = 0.01
# Randomly initialise U, V, F, b_m and b_n
U = np.random.normal(scale=5 / np.sqrt(latentDim), size=latentDim * M).reshape(
    M, latentDim
)
V = np.random.normal(scale=5 / np.sqrt(latentDim), size=latentDim * N).reshape(
    N, latentDim
)
# There are 20 unique genres
F = np.random.normal(size=latentDim * 20).reshape(20, latentDim)
b_m = np.zeros(M)
b_n = np.zeros(N)

# Start estimation algorithm - alternating least squares
tau_I_mat = tau * np.identity(latentDim)
conv = False
itr = 1
maxIter = 30
loglike = []
rmse_vals = []

while not conv:
    print(f"Iter = {itr}")
    dif = []
    # Loop over users in parallel
    for user in range(M):
        # User subset of user ratings
        user_ratings_subset = user_ratings[
            user_start_index[user] : user_end_index[user]
        ]
        num_movies = user_ratings_subset.shape[0]
        userBias = 0
        for row in user_ratings_subset:
            n = row[1]
            rmn = row[2]
            userBias += rmn - np.dot(U[user, :], V[n, :]) - b_n[n]
        userBias = lmd * userBias / (alpha + lmd * num_movies)
        # Compute difference in user bias
        dif.append(np.abs(userBias - b_m[user]))
        # Perform update
        b_m[user] = userBias
        # Compute user trait vector using cholesky decompostion
        ratings_term = np.zeros(V[n, :].shape)
        vv_mat = np.zeros(tau_I_mat.shape)
        for row in user_ratings_subset:
            n = row[1]
            rmn = row[2]
            ratings_term += (rmn - b_n[n] - b_m[user]) * V[n, :]
            vv_mat += np.matmul(
                V[n, :].reshape((latentDim, 1)), V[n, :].reshape((1, latentDim))
            )
        # Cholesky decomposition
        c, low = cho_factor(lmd * vv_mat + tau_I_mat)
        user_trait_vector = cho_solve(
            (c, low), lmd * ratings_term.reshape((latentDim, 1))
        ).reshape(U[user, :].shape)
        # Compute max difference in user trait vector
        dif.append(np.max(np.abs(user_trait_vector - U[user, :])))
        # Perform update
        U[user, :] = user_trait_vector

    # Loop over movies in parallel
    for movie in range(N):
        # Movie subset of movie ratings
        movie_ratings_subset = movie_ratings[
            movie_start_index[movie] : movie_end_index[movie]
        ]
        # Number of users that rated the movie
        num_users = movie_ratings_subset.shape[0]
        movieBias = 0
        for row in movie_ratings_subset:
            m = row[1]
            rmn = row[2]
            # Compute movie bias
            movieBias = rmn - np.dot(V[movie, :], U[m, :]) - b_m[m]
        movieBias = lmd * movieBias / (alpha + lmd * num_users)
        # compute difference in movie bias
        dif.append(np.abs(movieBias - b_n[movie]))
        # perform update
        b_n[movie] = movieBias
        # Compute movie trait vector using cholesky decompostion
        ratings_term = np.zeros(U[m, :].shape)
        uu_mat = np.zeros(tau_I_mat.shape)
        for row in movie_ratings_subset:
            m = row[1]
            rmn = row[2]
            ratings_term += (rmn - b_n[movie] - b_m[m]) * U[m, :]
            uu_mat += np.matmul(
                U[m, :].reshape((latentDim, 1)), U[m, :].reshape((1, latentDim))
            )
        # Feature component
        feat = (tau / np.sqrt(movie_dict[movie]["genre_count"])) * F[
            movie_dict[movie]["genre_values"], :
        ].sum(axis=0)
        # Cholesky decomposition
        c, low = cho_factor(lmd * uu_mat + tau_I_mat)
        movie_trait_vector = cho_solve(
            (c, low),
            lmd * ratings_term.reshape((latentDim, 1)) + feat.reshape((latentDim, 1)),
        ).reshape(V[movie, :].shape)
        # Compute max difference in movie trait vector
        dif.append(np.max(np.abs(movie_trait_vector - V[movie, :])))
        # Perform update
        V[movie, :] = movie_trait_vector

    # Loop over features
    denominator = 0
    numerator = 0
    for key, movies in genre_dict.items():
        # For each genre, get the list of movies
        for movie in movies:
            # sum of a movie's feature vectors, without the genre in question
            genres_list = [g for g in movie_dict[movie]["genre_values"] if g != key]
            sum_feature_vectors = F[genres_list, :].sum(axis=0)
            numerator += (
                V[movie, :]
                - sum_feature_vectors / np.sqrt(movie_dict[movie]["genre_count"])
            ) / np.sqrt(movie_dict[movie]["genre_count"])
            denominator += 1 / movie_dict[movie]["genre_count"]
        numerator *= tau
        denominator = beta + tau * denominator
        feature_vector = numerator / denominator
        # Compute max difference in feature vector
        dif.append(np.max(np.abs(feature_vector - F[key, :])))
        # Perform update
        F[key, :] = feature_vector

    # Print out loglikelihoods
    loglike.append(
        reg_logll_with_genre(
            U,
            V,
            F,
            tau,
            alpha,
            beta,
            lmd,
            b_m,
            b_n,
            M,
            user_ratings,
            user_start_index,
            user_end_index,
            movie_dict,
        )
    )
    print(f"LL = {loglike[-1]}")

    # RMSE
    rmse_vals.append(rmse(U, V, b_m, b_n, user_ratings))
    print(f"RMSE = {rmse_vals[-1]}")
    print(f"Maximum dif in {np.max(dif)}")

    # Save parameters, log-likelihood and rmse every 5 iterations
    if itr % 5 == 0:
        with open(f"explicit_ratings/param/u_matrix_{itr}_genre.npy", "wb") as f:
            np.save(f, U)
        with open(f"explicit_ratings/param/v_matrix_{itr}_genre.npy", "wb") as f:
            np.save(f, V)
        with open(f"explicit_ratings/param/bias_user_{itr}_genre.npy", "wb") as f:
            np.save(f, b_m)
        with open(f"explicit_ratings/param/bias_movie_{itr}_genre.npy", "wb") as f:
            np.save(f, b_n)
        with open("explicit_ratings/param/loglik_genre.npy", "wb") as f:
            np.save(f, np.array(loglike))
        with open("explicit_ratings/param/rmse_vals_genre.npy", "wb") as f:
            np.save(f, np.array(rmse_vals))

    if (np.max(dif) < 0.05) or (itr == maxIter):
        conv = True
        if np.max(dif) < 0.05:
            print(f"Convergence achieved at iteration {itr}.")
        else:
            print(f"Maximum number of iterations reached at {itr}.")
        print(f"Final regularised log likelihood = {loglike[-1]}")
        print(f"RMSE = {rmse_vals[-1]}")
        print(f"Maximum change in {np.max(dif)}")

    # Increment iterations
    itr += 1

    gc.collect()

# Save final parameters
with open("explicit_ratings/param/u_matrix_final_genre.npy", "wb") as f:
    np.save(f, U)
with open("explicit_ratings/param/v_matrix_final_genre.npy", "wb") as f:
    np.save(f, V)
with open("explicit_ratings/param/bias_user_final_genre.npy", "wb") as f:
    np.save(f, b_m)
with open("explicit_ratings/param/bias_movie_final_genre.npy", "wb") as f:
    np.save(f, b_n)
with open("explicit_ratings/param/loglik_genre.npy", "wb") as f:
    np.save(f, np.array(loglike))
with open("explicit_ratings/param/rmse_vals_genre.npy", "wb") as f:
    np.save(f, np.array(rmse_vals))
