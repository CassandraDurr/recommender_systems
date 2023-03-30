"""Functions required for prac 1, recommender system with explicit rating data."""
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve


def create_ratings_df(file_name: str) -> pd.DataFrame:
    """Ingest a ratings csv with columns userId, movieId and ratings and adapt indices.

    Args:
        file_name (str): Location of ratings csv to be ingested.

    Returns:
        pd.DataFrame: A dataframe with:
            - userId: user identifiers starting at 0 runing over consecutive values.
            - movieId: movie identifiers starting at 0.
            - movieId_order: movie identifiers starting at 0 runing over consecutive values.
            - rating_10: 10 scale rating instead of 5 stars, with half ratings.
    """
    ratings = pd.read_csv(file_name)
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
    id_shift = pd.DataFrame()
    id_shift["movieId"] = ratings["movieId"].unique().copy()
    id_shift = id_shift.sort_values(by="movieId")
    id_shift.reset_index(drop=True, inplace=True)
    id_shift.reset_index(drop=False, inplace=True)
    id_shift.columns = ["movieId_order", "movieId"]

    # Combine dataframes on "movieId"
    ratings = pd.merge(ratings, id_shift)
    return ratings


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
        -(alpha / 2) * np.matmul(bias_user.T, bias_user)
        - (alpha / 2) * np.matmul(bias_movie.T, bias_movie)
    )
    # U, Ut
    val = 0
    for row in u_mat:
        val += np.dot(row, row)
    log_lik = log_lik - (tau / 2) * val
    # V, Vt
    val = 0
    for row in v_mat:
        val += np.dot(row, row)
    log_lik = log_lik - (tau / 2) * val
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
    tau_identity_mat: np.array,
    latent_dim: int,
    u_matrix: np.array,
    v_matrix: np.array,
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
        tau_identity_mat (np.array): Trait vector regulariser multiplied by a identity matrix of shape (latent_dim, latent_dim).
        latent_dim (int): Latent dimension of user and movie trait vectors.
        u_matrix (np.array): User matrix.
        v_matrix (np.array): Movie matrix.
        b_n (np.array): Bias vector for movies.
        b_m (np.array): Bias vector for users.

    Returns:
        tuple[np.array, np.array]: Updated U, b_m.
    """
    copied_u = u_matrix.copy()
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
            n_mov = row[1]
            rmn = row[2]
            user_bias += (
                rmn - np.dot(copied_u[user, :], v_matrix[n_mov, :]) - b_n[n_mov]
            )
        user_bias = lmd * user_bias / (alpha + lmd * num_movies)
        # Perform update
        copied_b_m[user] = user_bias
        # Compute user trait vector using cholesky decompostion
        ratings_term = np.zeros(v_matrix[n_mov, :].shape)
        vv_mat = np.zeros(tau_identity_mat.shape)
        for row in user_ratings_subset:
            n_mov = row[1]
            rmn = row[2]
            ratings_term += (rmn - b_n[n_mov] - copied_b_m[user]) * v_matrix[n_mov, :]
            vv_mat += np.matmul(
                v_matrix[n_mov, :].reshape((latent_dim, 1)),
                v_matrix[n_mov, :].reshape((1, latent_dim)),
            )
        # Cholesky decomposition
        c, low = cho_factor(lmd * vv_mat + tau_identity_mat)
        user_trait_vector = cho_solve(
            (c, low), lmd * ratings_term.reshape((latent_dim, 1))
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
    tau_identity_mat: np.array,
    latent_dim: int,
    u_matrix: np.array,
    v_matrix: np.array,
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
        tau_identity_mat (np.array): Trait vector regulariser multiplied by a identity matrix of shape (latent_dim, latent_dim).
        latent_dim (int): Latent dimension of user and movie trait vectors.
        u_matrix (np.array): User matrix.
        v_matrix (np.array): Movie matrix.
        b_n (np.array): Bias vector for movies.
        b_m (np.array): Bias vector for users.

    Returns:
        tuple[np.array, np.array]: Updated V, b_n.
    """
    copied_v = v_matrix.copy()
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
            m_user = row[1]
            rmn = row[2]
            # Compute movie bias
            movie_bias = (
                rmn - np.dot(copied_v[movie, :], u_matrix[m_user, :]) - b_m[m_user]
            )
        movie_bias = lmd * movie_bias / (alpha + lmd * num_users)
        # perform update
        copied_b_n[movie] = movie_bias
        # Compute movie trait vector using cholesky decompostion
        ratings_term = np.zeros(u_matrix[m_user, :].shape)
        uu_mat = np.zeros(tau_identity_mat.shape)
        for row in movie_ratings_subset:
            m_user = row[1]
            rmn = row[2]
            ratings_term += (rmn - copied_b_n[movie] - b_m[m_user]) * u_matrix[
                m_user, :
            ]
            uu_mat += np.matmul(
                u_matrix[m_user, :].reshape((latent_dim, 1)),
                u_matrix[m_user, :].reshape((1, latent_dim)),
            )
        # Cholesky decomposition
        c, low = cho_factor(lmd * uu_mat + tau_identity_mat)
        movie_trait_vector = cho_solve(
            (c, low), lmd * ratings_term.reshape((latent_dim, 1))
        ).reshape(copied_v[movie, :].shape)
        # Perform update
        copied_v[movie, :] = movie_trait_vector

    return copied_v, copied_b_n
