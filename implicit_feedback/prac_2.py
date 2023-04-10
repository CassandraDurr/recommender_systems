"""Functions required for prac 2, recommender system with explicit rating data."""
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions import BayesianPersonalisedRanking, index_data, create_ratings_df

gc.collect()

# Combine dataframes on "movieId"
ratings = create_ratings_df(file_name="ratings_small.csv")

# # Keep aside a set of observations for testing
# train_ratings = ratings.sample(frac = 0.9, random_state=441998)
# test_ratings = ratings.loc[~ratings.index.isin(train_ratings.index)]

# # Check that there are no users or movies only in test
# print(set(test_ratings["userId"].unique()).difference(set(train_ratings["userId"].unique())))
# print(set(test_ratings["movieId_order"].unique()).difference(set(train_ratings["movieId_order"].unique())))

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
movie_frequencies = np.array(movie_frequencies.frequency)

# Number of users and movies
# M=162541, N=59047
num_users = ratings["userId"].nunique()
num_movies = ratings["movieId"].nunique()

# Obtain user index for training
user_idx, user_start_index, user_end_index = index_data(
    ratings, "rating_10", "userId", "movieId_order"
)
# Drop the rating column
user_idx = user_idx[:, 0:2]

# Delete ratings from memory to save some space now that all operations have been done with it
del ratings
gc.collect()

# Movie genres df
movie_genres = pd.read_csv(
    "implicit_feedback/movies_small_genres.csv", converters={"genres_v2": pd.eval}
)

# Assumptions and initialisations
latentDim = 12
lmd = 0.1
tau = 0.01
alpha = 0.01
U = np.random.normal(scale=5 / np.sqrt(latentDim), size=latentDim * num_users).reshape(
    num_users, latentDim
)
V = np.random.normal(scale=5 / np.sqrt(latentDim), size=latentDim * num_movies).reshape(
    num_movies, latentDim
)

# Set up class
BPR = BayesianPersonalisedRanking(
    user_matrix_u=U,
    movie_matrix_v=V,
    num_items=num_movies,
    num_users=num_users,
    user_idx=user_idx,
    user_start_index=user_start_index,
    user_end_index=user_end_index,
    latent_dim=latentDim,
    learning_rate=0.01,
)

conv = False
itr = 1
maxIter = 30
recall_at_k = []
precision_at_k = []

while not conv:
    print(f"Iter {itr}")
    # Shuffle users
    user_order = np.arange(num_users)
    np.random.shuffle(user_order)

    for user in user_order:
        user_subset = BPR.user_idx[
            BPR.user_start_index[user] : BPR.user_end_index[user]
        ]
        positive_movies = user_subset[:, 1]
        for movie in positive_movies:
            # neg = BPR.sample_negative_per_user_naive(
            #     user=user, movie_frequencies=movie_frequencies
            # )
            neg = BPR.sample_negative_per_user_genre(
                user=user,
                movie_frequencies=movie_frequencies,
                pos_movie=movie,
                genre_info=movie_genres,
            )
            triplet = [user, movie, neg]
            x_uij = BPR.predict(user=triplet[0], movie=triplet[1]) - BPR.predict(
                user=triplet[0], movie=triplet[2]
            )
            gradients = BPR.compute_gradients(triplet=triplet, x_uij=x_uij)
            # Perform update
            BPR.sgd_update(triplet=triplet, gradients=gradients, regulariser=0.01)

    # Save parameters every 10 iterations
    if itr % 5 == 0:
        with open(f"implicit_feedback/param/u_matrix_{itr}.npy", "wb") as f:
            np.save(f, BPR.user_matrix_u)
        with open(f"implicit_feedback/param/v_matrix_{itr}.npy", "wb") as f:
            np.save(f, BPR.movie_matrix_v)

    # Find the average precision and recall at k
    avg_precision_at_k, avg_recall_at_k = BPR.precision_and_recall_at_k(k=6)
    print(f"Precision = {avg_precision_at_k}, Recall = {avg_recall_at_k}")
    precision_at_k.append(avg_precision_at_k)
    recall_at_k.append(avg_recall_at_k)

    if itr == maxIter:
        conv = True
        print(f"Maximum number of iterations reached at {itr}.")

    # Increment iterations
    itr += 1

    gc.collect()

# Save parameters
with open("implicit_feedback/param/u_matrix_final.npy", "wb") as f:
    np.save(f, BPR.user_matrix_u)
with open("implicit_feedback/param/v_matrix_ifinal.npy", "wb") as f:
    np.save(f, BPR.movie_matrix_v)

# Plot the average precision and recall over the iterations
plt.figure(figsize=(9, 6))
plt.plot(precision_at_k, label="Avg precision at k")
plt.plot(recall_at_k, label="Avg recall at k")
plt.xlabel("Iteration")
plt.legend()
plt.savefig("implicit_feedback/figures/precision_recall.png")
plt.show()
