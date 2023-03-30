"""Functions required for prac 1."""
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve


def reg_logll(
    u_mat: np.array,
    v_mat: np.array,
    tau: float,
    alpha: float,
    lmd: float,
    bias_user: np.array,
    bias_movie: np.array,
    num_users: int,
    user_ratings: np.array,
    user_start_index: list,
    user_end_index: list,
) -> float:
    """Return th regularised log likelihood.

    Args:
        u_mat (np.array): User matrix, U.
        v_mat (np.array): Movie matrix, V.
        tau (float): Trait vectors regulariser.
        alpha (float): Bias regulariser.
        lmd (float): Regulariser.
        bias_user (np.array): Bias vector for users.
        bias_movie (np.array): Bias vector for movies.
        num_users (int): Number of users.
        user_ratings (np.array): Array of user ids, movie ids and ratings.
        user_start_index (list): Start index for users.
        user_end_index (list): End index for users.

    Returns:
        float: Regularised log likelihood.
    """
    log_lik = (
        # -(tau / 2) * np.sum(np.matmul(u_mat, u_mat.T).diagonal())
        # - (tau / 2) * np.sum(np.matmul(v_mat, v_mat.T).diagonal())
        - (alpha / 2) * np.matmul(bias_user.T, bias_user)
        - (alpha / 2) * np.matmul(bias_movie.T, bias_movie)
    )
    # U, Ut
    val = 0
    for row in u_mat:
        val += np.dot(row, row)
    log_lik = log_lik -(tau / 2) * val
    # V, Vt
    val = 0
    for row in v_mat:
        val += np.dot(row, row)
    log_lik = log_lik -(tau / 2) * val
    # term with m and n element of omega(m)
    term = 0
    for user in range(num_users):
        # User subset of user ratings
        user_ratings_subset = user_ratings[
            user_start_index[user] : user_end_index[user]
        ]
        for row in user_ratings_subset:
            n_movie = row[1]
            rmn = row[2]
            term += np.square(
                rmn
                - (
                    np.dot(u_mat[user, :], v_mat[n_movie, :])
                    + bias_movie[n_movie]
                    + bias_user[user]
                )
            )
    # Now update negative loglikelihood
    log_lik = -(lmd / 2) * term - log_lik
    return log_lik


def index_data(
    ratings_df: pd.DataFrame,
    rating_col_name: str,
    id_col_name: str,
    other_col_name: str,
) -> tuple[np.array, list, list]:
    """Produce a long sorted list of id and rating pairs,
    as well and start and end indices that are used to
    navigate the long sorted list easily.

    Args:
        ratings_df (pd.DataFrame): Dataframe containing user and item ids as well as the corresponding ratings.
        rating_col_name (str): Dataframe column name for the ratings.
        id_col_name (str): Dataframe column name for the id.
        other_col_name (str): Dataframe column name for the other id.

    Returns:
        tuple[np.array, list, list]: Sorted list of id and rating pairs, start index and end index.
    """
    # Sort ratings data by id
    id_ratings = (
        ratings_df[[id_col_name, other_col_name, rating_col_name]]
        .sort_values(by=[id_col_name, other_col_name])
        .reset_index(drop=True)
    )
    # Obtain frequencies per id
    id_ratings_index = (
        id_ratings[id_col_name]
        .value_counts()
        .reset_index()
        .sort_values(by="index")
        .reset_index(drop=True)
    )
    id_ratings_index.columns = [id_col_name, "frequency"]
    # start and end index
    start_index = np.cumsum(id_ratings_index["frequency"])
    start_index = start_index.tolist()
    start_index.insert(0, 0)  # add a zero to the start of the list
    start_index = start_index[:-1]  # remove the last element of the list
    end_index = np.cumsum(id_ratings_index["frequency"]).tolist()
    # structure for id and ratings = np array
    # Np arrays can be more compact than tuples
    id_ratings = np.array(id_ratings, dtype=int)

    return id_ratings, start_index, end_index


def rmse(
    u_mat: np.array,
    v_mat: np.array,
    bias_user: np.array,
    bias_movie: np.array,
    user_ratings: np.array,
) -> float:
    """Return the root mean squared error.

    Args:
        u_mat (np.array): User matrix, U.
        v_mat (np.array): Movie matrix, V.
        bias_user (np.array): Bias vector for users.
        bias_movie (np.array): Bias vector for movies.
        user_ratings (np.array): Array of user ids, movie ids and ratings.

    Returns:
        float: RMSE value.
    """
    squared_error = 0
    num_ratings = user_ratings.shape[0]
    for row in user_ratings:
        m_user, n_movie, rmn = row[0], row[1], row[2]
        squared_error += np.square(
            np.dot(u_mat[m_user, :], v_mat[n_movie, :])
            + bias_user[m_user]
            + bias_movie[n_movie]
            - rmn
        )
    rmse_result = np.sqrt((1 / num_ratings) * squared_error)
    return rmse_result


def update_user(
    start: int,
    end: int,
    user_ratings: np.array,
    user_start_index: list,
    user_end_index: list,
    lmd: float,
    alpha: float,
    tau_I_mat: np.array,
    latentDim: int,
    U: np.array,
    V: np.array,
    b_n: np.array,
    b_m: np.array,
) -> tuple[np.array, np.array]:
    """Perform the update to a select number of users, for multiprocessing.

    Args:
        start (int): Start index of users.
        end (int): End index of users.
        user_ratings (np.array): User ratings, {m,n,rmn}.
        user_start_index (list): User ratings start index.
        user_end_index (list): User ratings end index.
        lmd (float): Loss regulariser.
        alpha (float): Movie and user bias regulariser.
        tau_I_mat (np.array): Trait vector regulariser multiplied by a identity matrix of shape (latentDim, latentDim).
        latentDim (int): Latent dimension of user and movie trait vectors.
        U (np.array): User matrix.
        V (np.array): Movie matrix.
        b_n (np.array): Bias vector for movies.
        b_m (np.array): Bias vector for users.

    Returns:
        tuple[np.array, np.array]: Updated U, b_m.
    """
    copied_u = U.copy()
    copied_u.flags.writeable = True
    copied_b_m = b_m.copy()
    copied_b_m.flags.writeable = True
    for user in range(start, end):
        # User subset of user ratings
        user_ratings_subset = user_ratings[
            user_start_index[user] : user_end_index[user]
        ]
        num_movies = user_ratings_subset.shape[0]
        user_bias = 0
        for row in user_ratings_subset:
            n = row[1]
            rmn = row[2]
            user_bias += rmn - np.dot(copied_u[user, :], V[n, :]) - b_n[n]
        user_bias = lmd * user_bias / (alpha + lmd * num_movies)
        # Perform update
        copied_b_m[user] = user_bias
        # Compute user trait vector using cholesky decompostion
        ratings_term = np.zeros(V[n, :].shape)
        vv_mat = np.zeros(tau_I_mat.shape)
        for row in user_ratings_subset:
            n = row[1]
            rmn = row[2]
            ratings_term += (rmn - b_n[n] - copied_b_m[user]) * V[n, :]
            vv_mat += np.matmul(
                V[n, :].reshape((latentDim, 1)), V[n, :].reshape((1, latentDim))
            )
        # Cholesky decomposition
        c, low = cho_factor(lmd * vv_mat + tau_I_mat)
        user_trait_vector = cho_solve(
            (c, low), lmd * ratings_term.reshape((latentDim, 1))
        ).reshape(copied_u[user, :].shape)
        # Perform update
        copied_u[user, :] = user_trait_vector
    
    return copied_u, copied_b_m

        


def update_movie(
    start: int,
    end: int,
    movie_ratings: np.array,
    movie_start_index: list,
    movie_end_index: list,
    lmd: float,
    alpha: float,
    tau_I_mat: np.array,
    latentDim: int,
    U: np.array,
    V: np.array,
    b_n: np.array,
    b_m: np.array,
) -> tuple[np.array, np.array]:
    """Perform the update to a select number of users, for multiprocessing.

    Args:
        start (int): Start index of users.
        end (int): End index of users.
        movie_ratings (np.array): Movie ratings, {n,m,rmn}.
        movie_start_index (list): Movie ratings start index.
        movie_end_index (list): Movie ratings end index.
        lmd (float): Loss regulariser.
        alpha (float): Movie and user bias regulariser.
        tau_I_mat (np.array): Trait vector regulariser multiplied by a identity matrix of shape (latentDim, latentDim).
        latentDim (int): Latent dimension of user and movie trait vectors.
        U (np.array): User matrix.
        V (np.array): Movie matrix.
        b_n (np.array): Bias vector for movies.
        b_m (np.array): Bias vector for users.

    Returns:
        tuple[np.array, np.array]: Updated V, b_n.
    """
    copied_v = V.copy()
    copied_v.flags.writeable = True
    copied_b_n = b_n.copy()
    copied_b_n.flags.writeable = True
    # Loop over movies in parallel
    for movie in range(start, end):
        # Movie subset of movie ratings
        movie_ratings_subset = movie_ratings[
            movie_start_index[movie] : movie_end_index[movie]
        ]
        # Number of users that rated the movie
        num_users = movie_ratings_subset.shape[0]
        movie_bias = 0
        for row in movie_ratings_subset:
            m = row[1]
            rmn = row[2]
            # Compute movie bias
            movie_bias = rmn - np.dot(copied_v[movie, :], U[m, :]) - b_m[m]
        movie_bias = lmd * movie_bias / (alpha + lmd * num_users)
        # perform update
        copied_b_n[movie] = movie_bias
        # Compute movie trait vector using cholesky decompostion
        ratings_term = np.zeros(U[m, :].shape)
        uu_mat = np.zeros(tau_I_mat.shape)
        for row in movie_ratings_subset:
            m = row[1]
            rmn = row[2]
            ratings_term += (rmn - copied_b_n[movie] - b_m[m]) * U[m, :]
            uu_mat += np.matmul(
                U[m, :].reshape((latentDim, 1)), U[m, :].reshape((1, latentDim))
            )
        # Cholesky decomposition
        c, low = cho_factor(lmd * uu_mat + tau_I_mat)
        movie_trait_vector = cho_solve(
            (c, low), lmd * ratings_term.reshape((latentDim, 1))
        ).reshape(copied_v[movie, :].shape)
        # Perform update
        copied_v[movie, :] = movie_trait_vector
    
    return copied_v, copied_b_n
        

