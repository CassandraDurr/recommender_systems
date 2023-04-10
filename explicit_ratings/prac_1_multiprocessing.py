"""Module to build recommender system from prac 1, with multiprocessing."""
import gc
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions import (
    reg_logll,
    rmse,
    index_data,
    update_user,
    update_movie,
)

gc.collect()


ratings = pd.read_csv("ratings_25m.csv")
ratings = ratings.drop(columns="timestamp")
# use 1 to 10 scale to work in integers
ratings["rating_10"] = ratings["rating"] * 2
# start the user and movie ratings at 0
ratings["userId"] = ratings["userId"] - 1
ratings["movieId"] = ratings["movieId"] - 1
print(
    f"User id: min={np.min(ratings['userId'])}, max = {np.max(ratings['userId'])}, total = {ratings['userId'].nunique()}"
)
print(
    f"Movie id: min={np.min(ratings['movieId'])}, max = {np.max(ratings['movieId'])}, total = {ratings['movieId'].nunique()}"
)
# There is an issue that not all movies are present in the ratings csv.
# The number of movie ids = 59,047 but the maximum movie id = 209170
# We need to add a column to the ratings dataframe that matches
# each movie id to a new id from 0 to 59,046.
idShift = pd.DataFrame()
idShift["movieId"] = ratings["movieId"].unique().copy()
idShift = idShift.sort_values(by="movieId")
idShift.reset_index(drop=True, inplace=True)
idShift.reset_index(drop=False, inplace=True)
idShift.columns = ["movieId_order", "movieId"]

# Combine dataframes on "movieId"
ratings = pd.merge(ratings, idShift)


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
# M = 162541, N = 59047 or
# M = 610, N = 9724
M = ratings["userId"].nunique()
print(f"M = {M}")
N = ratings["movieId"].nunique()
print(f"N = {N}")

# Delete ratings from memory to save some space now that all operations have been done with it
del ratings
del idShift
gc.collect()

lmd = 0.15
tau = 0.01  # related to latent dimension
alpha = 0.01
# Randomly initialise U, V, b_m and b_n
U = np.random.normal(scale=np.sqrt(5 / np.sqrt(latentDim)), size=latentDim * M).reshape(
    M, latentDim
)
V = np.random.normal(scale=np.sqrt(5 / np.sqrt(latentDim)), size=latentDim * N).reshape(
    N, latentDim
)
b_m = np.zeros(M)
b_n = np.zeros(N)

# Start estimation algorithm - alternating least squares
tau_I_mat = tau * np.identity(latentDim)
conv = False
itr = 1
maxIter = 100
loglike = []
rmse_vals = []

while not conv:
    print(f"Iter = {itr}")
    dif = []

    # Old values
    U_old = U.copy()
    V_old = V.copy()
    b_m_old = b_m.copy()
    b_n_old = b_n.copy()

    # M = 162541, N = 59047
    # User arguments
    start_itm = [
        0,
        13545,
        27090,
        40635,
        54180,
        67725,
        81270,
        94815,
        108360,
        121905,
        135450,
        148995,
    ]
    end_itm = [
        13545,
        27090,
        40635,
        54180,
        67725,
        81270,
        94815,
        108360,
        121905,
        135450,
        148995,
        M,
    ]
    # start_itm = [0, 61, 122, 183, 244, 305, 366, 427, 488, 549]
    # end_itm = [61, 122, 183, 244, 305, 366, 427, 488, 549, M]

    # Movie arguments
    start_itm_mov = [
        0,
        4920,
        9840,
        14760,
        19680,
        24600,
        29520,
        34440,
        39360,
        44280,
        49200,
        54120,
    ]
    end_itm_mov = [
        4920,
        9840,
        14760,
        19680,
        24600,
        29520,
        34440,
        39360,
        44280,
        49200,
        54120,
        N,
    ]
    # start_itm_mov = [0, 972, 1994, 2916, 3888, 4860, 5832, 6804, 7776, 8748]
    # end_itm_mov = [972, 1994, 2916, 3888, 4860, 5832, 6804, 7776, 8748, N]

    # Run code in parallel
    user_parallel = Parallel(n_jobs=12)(
        delayed(update_user)(
            start_itm[i],
            end_itm[i],
            user_ratings,
            user_start_index,
            user_end_index,
            lmd,
            alpha,
            tau_I_mat,
            latentDim,
            U,
            V,
            b_n,
            b_m,
        )
        for i in range(len(start_itm))
    )
    # Copy the updated arrays back to the original arrays
    for i, itm in enumerate(start_itm):
        U[itm : end_itm[i], :] = user_parallel[i][0][itm : end_itm[i], :]
        b_m[itm : end_itm[i]] = user_parallel[i][1][itm : end_itm[i]]

    # Run code in parallel
    movie_parallel = Parallel(n_jobs=12)(
        delayed(update_movie)(
            start_itm_mov[i],
            end_itm_mov[i],
            movie_ratings,
            movie_start_index,
            movie_end_index,
            lmd,
            alpha,
            tau_I_mat,
            latentDim,
            U,
            V,
            b_n,
            b_m,
        )
        for i in range(len(start_itm_mov))
    )

    # Copy the updated arrays back to the original arrays
    for i, itm in enumerate(start_itm_mov):
        V[itm : end_itm_mov[i], :] = movie_parallel[i][0][itm : end_itm_mov[i], :]
        b_n[itm : end_itm_mov[i]] = movie_parallel[i][1][itm : end_itm_mov[i]]

    # Compute the maximum differences
    dif.append(
        np.max(
            [
                np.max(np.abs(U - U_old)),
                np.max(np.abs(V - V_old)),
                np.max(np.abs(b_m - b_m_old)),
                np.max(np.abs(b_n - b_n_old)),
            ]
        )
    )

    # Increment iterations
    itr += 1

    # Log likelihoods
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

    # Save parameters
    if itr % 20 == 0:
        with open("param/u_matrix.npy", "wb") as f:
            np.save(f, U)
        with open("param/v_matrix.npy", "wb") as f:
            np.save(f, V)
        with open("param/bias_user.npy", "wb") as f:
            np.save(f, b_m)
        with open("param/bias_movie.npy", "wb") as f:
            np.save(f, b_n)
        # Plot log likelihood
        plt.figure(figsize=(9, 6))
        plt.plot(loglike, color="#1E63A4")
        plt.axhline(
            loglike[-1], color="#F05225", linestyle="dotted", label="Final LL value"
        )
        plt.xlabel("Iteration")
        plt.ylabel("Log likelihood")
        plt.title(f"Log likelihood over training iterations (iter = {loglike[-1]})")
        plt.savefig(f"figures/loglikelihood_{itr}.png")
        plt.show()

        # Plot RMSE
        plt.figure(figsize=(9, 6))
        plt.plot(rmse_vals, color="#1E63A4")
        plt.axhline(
            rmse_vals[-1], color="#F05225", linestyle="dotted", label="Final RMSE value"
        )
        plt.xlabel("Iteration")
        plt.ylabel("RMSE")
        plt.title(f"RMSE over training iterations (final = {rmse_vals[-1]})")
        plt.savefig(f"figures/RMSE_{itr}.png")
        plt.show()

    # Convergence
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

with open("param/u_matrix.npy", "wb") as f:
    np.save(f, U)
with open("param/v_matrix.npy", "wb") as f:
    np.save(f, V)
with open("param/bias_user.npy", "wb") as f:
    np.save(f, b_m)
with open("param/bias_movie.npy", "wb") as f:
    np.save(f, b_n)

# Plot log likelihood
plt.figure(figsize=(9, 6))
plt.plot(loglike, color="#1E63A4")
plt.axhline(loglike[-1], color="#F05225", linestyle="dotted", label="Final LL value")
plt.xlabel("Iteration")
plt.ylabel("Log likelihood")
plt.title(f"Log likelihood over training iterations (final = {loglike[-1]})")
plt.savefig("figures/loglikelihood.png")
plt.show()

# Plot RMSE
plt.figure(figsize=(9, 6))
plt.plot(rmse_vals, color="#1E63A4")
plt.axhline(
    rmse_vals[-1], color="#F05225", linestyle="dotted", label="Final RMSE value"
)
plt.xlabel("Iteration")
plt.ylabel("RMSE")
plt.title(f"RMSE over training iterations (final = {rmse_vals[-1]})")
plt.savefig("figures/RMSE.png")
plt.show()
