"""Functions required for prac 2, recommender system with explicit rating data."""
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from functions import BayesianPersonalisedRanking, index_data, create_ratings_df

gc.collect()

# Combine dataframes on "movieId"
# ratings = create_ratings_df(file_name="data/ratings_25m.csv")
ratings = create_ratings_df(file_name="data/ratings_small.csv")

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
    ratings, "rating_10", "userId_order", "movieId_order"
)
# Drop the rating column
user_idx = user_idx[:, 0:2]

# Delete ratings from memory to save some space now that all operations have been done with it
del ratings
gc.collect()

# Movie genres df
movie_genres = pd.read_csv(
    "data/movies_small_genres.csv", converters={"genres_v2": pd.eval}
)
# movie_genres = pd.read_csv(
#     "data/movies_25m_genres.csv", converters={"genres_v2": pd.eval}
# )
# Assumptions and initialisations
latentDim = 12
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
    learning_rate=0.02,
)

conv = False
itr = 1
maxIter = 30
# recall_at_k_10 = []
# precision_at_k_10 = []
# recall_at_k_20 = []
# precision_at_k_20 = []
# recall_at_k_100 = []
# precision_at_k_100 = []
recall_at_k_40 = []
precision_at_k_40 = []
recall_at_k_50 = []
precision_at_k_50 = []
recall_at_k_60 = []
precision_at_k_60 = []

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

    # Save parameters every 5 iterations
    if itr % 5 == 0:
        with open(f"implicit_feedback/param/u_matrix_{itr}.npy", "wb") as f:
            np.save(f, BPR.user_matrix_u)
        with open(f"implicit_feedback/param/v_matrix_{itr}.npy", "wb") as f:
            np.save(f, BPR.movie_matrix_v)

    # Compute precision and recall at k for 3 values of k in parallel
    k_values = [40, 50, 60]
    results = Parallel(n_jobs=3)(
        delayed(BPR.compute_precision_and_recall_at_k)(k) for k in k_values
    )
    # Extract the precision at k for each k value
    precision_at_k = [result[0] for result in results]
    precision_at_k_40.append(precision_at_k[0])
    precision_at_k_50.append(precision_at_k[1])
    precision_at_k_60.append(precision_at_k[2])
    # Extract the recall at k for each k value
    recall_at_k = [result[1] for result in results]
    recall_at_k_40.append(recall_at_k[0])
    recall_at_k_50.append(recall_at_k[1])
    recall_at_k_60.append(recall_at_k[2])

    if itr == maxIter:
        conv = True
        print(f"Maximum number of iterations reached at {itr}.")

    # Increment iterations
    itr += 1

    gc.collect()

# Save parameters
with open("implicit_feedback/param/u_matrix_final.npy", "wb") as f:
    np.save(f, BPR.user_matrix_u)
with open("implicit_feedback/param/v_matrix_final.npy", "wb") as f:
    np.save(f, BPR.movie_matrix_v)

# Plot the average precision and recall over the iterations
# k = 40
plt.figure(figsize=(9, 6))
plt.plot(precision_at_k_40, label="Avg precision at k")
plt.plot(recall_at_k_40, label="Avg recall at k")
plt.xlabel("Iteration")
plt.title("Recall and precision at k = 40")
plt.legend()
plt.savefig("implicit_feedback/figures/precision_recall_40.png")
plt.show()
# k = 50
plt.figure(figsize=(9, 6))
plt.plot(precision_at_k_50, label="Avg precision at k")
plt.plot(recall_at_k_50, label="Avg recall at k")
plt.xlabel("Iteration")
plt.title("Recall and precision at k = 50")
plt.legend()
plt.savefig("implicit_feedback/figures/precision_recall_50.png")
plt.show()
# k = 60
plt.figure(figsize=(9, 6))
plt.plot(precision_at_k_60, label="Avg precision at k")
plt.plot(recall_at_k_60, label="Avg recall at k")
plt.xlabel("Iteration")
plt.title("Recall and precision at k = 60")
plt.legend()
plt.savefig("implicit_feedback/figures/precision_recall_60.png")
plt.show()
