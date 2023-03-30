"""Functions required for prac 2, recommender system with explicit rating data."""
import numpy as np
import pandas as pd


class BayesianPersonalisedRanking:
    """Bayesian personalised ranking algorithm."""

    def __init__(
        self,
        user_matrix_u,
        movie_matrix_v,
        num_users,
        num_items,
        latent_dim,
        learning_rate,
        lmd_user,
        lmd_posupdates,
        lmd_neg_updates,
        user_idx,
        user_start_index,
        user_end_index,
    ):
        # Initialise parameters
        self.user_matrix_u = user_matrix_u
        self.movie_matrix_v = movie_matrix_v
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        # Regularisation parameters
        self.lmd_user = lmd_user
        self.lmd_posupdates = lmd_posupdates
        self.lmd_neg_updates = lmd_neg_updates
        # User/ movie indices and star, end arrays per user
        self.user_idx = user_idx
        self.user_start_index = user_start_index
        self.user_end_index = user_end_index

    def predict(self, user: np.array, movie: np.array):
        """Prediction formula for BPR - dot product between user trait vector and movie trait vector.

        Args:
            user (np.array): _description_
            movie (np.array): _description_

        Returns:
            np.array: Dot product of the user vector and movie vector.
        """
        return np.dot(self.user_matrix_u[user, :], self.movie_matrix_v[movie, :])

    def force_probability_sum(self, probs: np.array) -> np.array:
        """Force probability distribution to sum to one.

        Args:
            probs (np.array): Probabilities for movies appearing.

        Returns:
            np.array: Valid probability distribution.
        """
        if probs.sum() != 1.0:
            probs *= 1.0 / probs.sum()
        return probs

    def draw_triplet_per_user_naive(
        self, user: int, movie_frequencies: np.array
    ) -> np.array:
        """Naive sampling strategy to obtain positive and negative movie samples per user.

        The strategy is naive becuase it doesn't account for movie genres.

        Args:
            user (int): userId
            movie_frequencies (np.array): Array of probabilities of each movie occurring.

        Returns:
            np.array: Array with [u, i, j] per row.
        """
        # u = user, i = positive movie, j = negative movie
        user_subset = self.user_idx[
            self.user_start_index[user] : self.user_end_index[user]
        ]
        # Acceptable negtives
        neg_options = sorted(
            set(np.arange(self.num_items)).difference(set(user_subset[:, 1]))
        )
        # Probabilities corresponding to negatives
        movie_frequencies = self.force_probability_sum(movie_frequencies[neg_options])
        # Sample the same number of negatives as positives
        neg_samples = np.random.choice(
            neg_options, p=movie_frequencies, size=user_subset.shape[0], replace=False
        )
        # Resultant u, i, j samples
        triplet = np.hstack([user_subset, neg_samples])
        return triplet


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
    """Return the regularised log likelihood.

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
    # Now update loglikelihood
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
