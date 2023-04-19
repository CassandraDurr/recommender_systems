"""Find recommendations based on a single movie rated and explore Napolean Dynamite effect."""
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

# Load U, V, b_n, b_m
print("25 iterations")
with open("explicit_ratings/param/u_matrix_25.npy", "rb") as f:
    U_mat = np.load(f)
with open("explicit_ratings/param/v_matrix_25.npy", "rb") as f:
    V_mat = np.load(f)
with open("explicit_ratings/param/bias_movie_25.npy", "rb") as f:
    bias_movie = np.load(f)
with open("explicit_ratings/param/bias_user_25.npy", "rb") as f:
    bias_user = np.load(f)

# Get the raw ratings
ratings = pd.read_csv("ratings_25m.csv")
ratings = ratings.drop(columns="timestamp")
# use 1 to 10 scale to work in integers
ratings["rating_10"] = ratings["rating"] * 2
# start the user and movie ratings at 0
ratings["userId_lessone"] = ratings["userId"] - 1
ratings["movieId_lessone"] = ratings["movieId"] - 1
idShift = pd.DataFrame()
idShift["movieId_lessone"] = ratings["movieId_lessone"].unique().copy()
idShift = idShift.sort_values(by="movieId_lessone")
idShift.reset_index(drop=True, inplace=True)
idShift.reset_index(drop=False, inplace=True)
idShift.columns = ["movieId_order", "movieId_lessone"]
# Combine dataframes on "movieId_lessone"
ratings = pd.merge(ratings, idShift)

# movie_ids = ratings[["movieId", "movieId_order"]].drop_duplicates()
# movie_pd = pd.read_csv("movies.csv")
# movie_pd = movie_pd[["movieId", "title"]]
# movie_ids = pd.merge(movie_ids, movie_pd)
# movie_ids = movie_ids.sort_values(by="movieId")
movie_ids = pd.read_csv("movie_ids.csv")

# Assume user bias = 0

n = 18913  # "Hobbit: An Unexpected Journey, The (2012)"
print("Hobbit: An Unexpected Journey, The (2012)")
# n = 16939  # Birdemic: Shock and Terror (2010)
# print("Birdemic: Shock and Terror (2010)")
# n = 38382  # Eyes of Crystal (2004)
# print("Eyes of Crystal (2004)")
rmn = 10.0

# Compute user trait vector
lmd = 0.1
tau = 0.01
alpha = 0.01
latentDim = 12
tau_I_mat = tau * np.identity(latentDim)
b_n = bias_movie
b_m = bias_user
ratings_term = (rmn - b_n[n] - 0) * V_mat[n, :]
vv_mat = np.matmul(
    V_mat[n, :].reshape((latentDim, 1)), V_mat[n, :].reshape((1, latentDim))
)
c, low = cho_factor(lmd * vv_mat + tau_I_mat)
user_trait_vector = cho_solve(
    (c, low), lmd * ratings_term.reshape((latentDim, 1))
).reshape((latentDim,))
# Compute score for each movie
score_for_movie = np.zeros((59047,))
for movie in range(59047):
    score_for_movie[movie] = (
        np.dot(user_trait_vector, V_mat[movie, :]) + 0.05 * bias_movie[movie]
    )
movieScores = pd.DataFrame()
movieScores["Scores"] = score_for_movie
movieScores = movieScores.reset_index()
movieScores.columns = ["movieId_order", "Scores"]
movieScores = pd.merge(movieScores, movie_ids)
movieScores = movieScores.sort_values(by="Scores", ascending=False)
print(movieScores.head(20))

# Remove movies that were only rated by a few people
removeMoviesLimit = 90
movie_id_exclude = list(
    ratings["movieId_order"]
    .value_counts()
    .loc[lambda x: x < removeMoviesLimit]
    .to_frame()
    .index
)
print(f"Excluded movies = {len(movie_id_exclude)}.")
movieScoresExclude = movieScores[~movieScores["movieId_order"].isin(movie_id_exclude)]
print(movieScoresExclude.head(20))

# Get the top 20 highest rated movies
# Exclude unpopular movies
top_ratings = ratings.groupby(["movieId_order"])["rating_10"].mean()
top_ratings = top_ratings.reset_index()
top_ratings = top_ratings.sort_values(by="rating_10", ascending=False)
top_ratings = top_ratings[~top_ratings["movieId_order"].isin(movie_id_exclude)]
print(top_ratings.head(20))

# # Napolean Dynamite Effect
# # Longest vectors
# movie_lens = np.linalg.norm(V_mat, axis = 1)
# print(np.argmax(movie_lens)) # Most polarising
# print(np.argmin(movie_lens)) # Least informative/ polarising
# indices_largest = np.argpartition(movie_lens, -5)[-5:]
# largest_lengths = movie_lens[indices_largest]
# indices_smallest = np.argpartition(movie_lens, 5)[:5]
# smallest_lengths = movie_lens[indices_smallest]
# print("\nLargest movie trait vectors\n", indices_largest)
# print("\nSmallest movie trait vectors\n", indices_smallest)
