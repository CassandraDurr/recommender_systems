"""Module to build recommender system from prac 1, with bias updates only."""
import gc
import numpy as np
import matplotlib.pyplot as plt
from functions import reg_logll, rmse, index_data, create_ratings_df

gc.collect()

# Combine dataframes on "movieId"
ratings = create_ratings_df(file_name="data/ratings_25m.csv")


user_ratings, user_start_index, user_end_index = index_data(
    ratings, "rating_10", "userId", "movieId_order"
)
movie_ratings, movie_start_index, movie_end_index = index_data(
    ratings, "rating_10", "movieId_order", "userId"
)


# Both user_ratings and movie_ratings are shape (25000095, 2)
# The first column is the userId/movieId and the second column is the ratings
# user_start_index and user_end_index have length M
# movie_start_index and movie_end_index have length N

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
# Save the initial user and movie trait vectors
with open("explicit_ratings/param/u_matrix_bias.npy", "wb") as f:
    np.save(f, U)
with open("explicit_ratings/param/v_matrix_bias.npy", "wb") as f:
    np.save(f, V)
# Initialise biases
b_m = np.zeros(M)
b_n = np.zeros(N)

# Start estimation algorithm - alternating least squares
tau_I_mat = tau * np.identity(latentDim)
conv = False
itr = 1
maxIter = 50
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
        # No user trait vector updates

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
        # No movie trait vectors update

    # Increment iterations
    itr += 1

    # Print out loglikelihoods
    loglike.append(
        reg_logll(
            U,
            V,
            tau,
            alpha,
            lmd,
            b_m,
            b_n,
            M,
            user_ratings,
            user_start_index,
            user_end_index,
        )
    )
    print(f"LL = {loglike[-1]}")

    # RMSE
    rmse_vals.append(rmse(U, V, b_m, b_n, user_ratings))
    print(f"RMSE = {rmse_vals[-1]}")
    print(f"Maximum dif in {np.max(dif)}")

    # Save parameters every 5 iterations
    if itr % 5 == 0:
        with open(f"explicit_ratings/param/bias_user_{itr}_bias.npy", "wb") as f:
            np.save(f, b_m)
        with open(f"explicit_ratings/param/bias_movie_{itr}_bias.npy", "wb") as f:
            np.save(f, b_n)
        # Store log likelihood and rmse
        with open("explicit_ratings/param/loglik_bias.npy", "wb") as f:
            np.save(f, np.array(loglike))
        with open("explicit_ratings/param/rmse_vals_bias.npy", "wb") as f:
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

    gc.collect()

# Save final parameters
with open("explicit_ratings/param/bias_user_bias.npy", "wb") as f:
    np.save(f, b_m)
with open("explicit_ratings/param/bias_movie_bias.npy", "wb") as f:
    np.save(f, b_n)
# Store log likelihood and rmse
with open("explicit_ratings/param/loglik_bias.npy", "wb") as f:
    np.save(f, np.array(loglike))
with open("explicit_ratings/param/rmse_vals_bias.npy", "wb") as f:
    np.save(f, np.array(rmse_vals))

# Plot log likelihood
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(loglike, color="#1E63A4")
plt.axhline(loglike[-1], color="#F05225", linestyle="dotted", label="Final LL value")
ax.ticklabel_format(useOffset=False)
plt.xlabel("Iteration")
plt.ylabel("Log likelihood")
plt.title("Log likelihood over training iterations, bias updates only")
plt.savefig("explicit_ratings/figures/loglikelihood.png")
plt.show()

# Plot RMSE
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(rmse_vals, color="#1E63A4")
plt.axhline(
    rmse_vals[-1], color="#F05225", linestyle="dotted", label="Final RMSE value"
)
ax.ticklabel_format(useOffset=False)
plt.xlabel("Iteration")
plt.ylabel("RMSE")
plt.title("RMSE over training iterations, bias updates only")
plt.savefig("explicit_ratings/figures/RMSE.png")
plt.show()
