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
    log_lik = log_lik - (lmd / 2) * term
    return log_lik


def reg_logll_with_genre(
    u_mat: np.array,
    v_mat: np.array,
    f_mat: np.array,
    tau: float,
    alpha: float,
    beta: float,
    lmd: float,
    bias_user: np.array,
    bias_movie: np.array,
    num_users: int,
    user_ratings: np.array,
    user_start_index: list,
    user_end_index: list,
    movie_dict: dict,
) -> float:
    """Return the regularised log likelihood, with genre information.

    Args:
        u_mat (np.array): User matrix, U.
        v_mat (np.array): Movie matrix, V.
        f_mat (np.array): Features matrix, F.
        tau (float): Trait vectors regulariser.
        alpha (float): Bias regulariser.
        beta (float): Feature vector regulariser.
        lmd (float): Regulariser.
        bias_user (np.array): Bias vector for users.
        bias_movie (np.array): Bias vector for movies.
        num_users (int): Number of users.
        user_ratings (np.array): Array of user ids, movie ids and ratings.
        user_start_index (list): Start index for users.
        user_end_index (list): End index for users.
        movie_dict (dict): Dictionary with movieId_order as key and genre info as items.

    Returns:
        float: Regularised log likelihood, including movie genre information.
    """
    log_lik = -(alpha / 2) * np.matmul(bias_user.T, bias_user) - (
        alpha / 2
    ) * np.matmul(bias_movie.T, bias_movie)
    # U, Ut
    val = 0
    for row in u_mat:
        val += np.dot(row, row)
    log_lik = log_lik - (tau / 2) * val
    # F, Ft
    val = 0
    for row in f_mat:
        val += np.dot(row, row)
    log_lik = log_lik - (beta / 2) * val
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
    log_lik = log_lik - (lmd / 2) * term
    # Term with features and movie trait vectors
    val = 0
    for key in movie_dict:
        # For each movie
        sum_feature_vectors = f_mat[movie_dict[key]["genre_values"], :].sum(axis=0)
        vector = v_mat[key, :] - sum_feature_vectors / np.sqrt(
            movie_dict[key]["genre_count"]
        )
        val += np.dot(vector, vector)
    log_lik = log_lik - (tau / 2) * val

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
        chol_factor, low = cho_factor(lmd * vv_mat + tau_identity_mat)
        user_trait_vector = cho_solve(
            (chol_factor, low), lmd * ratings_term.reshape((latent_dim, 1))
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
        chol_factor, low = cho_factor(lmd * uu_mat + tau_identity_mat)
        movie_trait_vector = cho_solve(
            (chol_factor, low), lmd * ratings_term.reshape((latent_dim, 1))
        ).reshape(copied_v[movie, :].shape)
        # Perform update
        copied_v[movie, :] = movie_trait_vector

    return copied_v, copied_b_n


def map_genres(genre_list: list[str], genres_dict: dict[str, int]) -> list[str]:
    """A function that can be used to map genre names as strings to indices for genres.

    Args:
        genre_list (list[str]): A list of genre names to be converted to integers (indices).
        genres_dict (dict[str, int]): Dictionary with genre names as keys and genre indices as items.

    Returns:
        list[str]: A list of genre indices converted from strings using a mapping.
    """
    return [genres_dict[genre] for genre in genre_list]


def genre_key_dict(file_location: str, genres: dict[str, int]) -> dict[int, list[int]]:
    """Produce a dictionary with genre id as key and movie ids as items.

    Args:
        file_location (str): Location of genres csv.
        genres (dict[str, int]): A dictionary relating genre names to genre ids.

    Returns:
        dict[int, list[int]]: Dictionary with genre as key and movies as items.

    """
    movie_genres = pd.read_csv(file_location, converters={"genres_v2": pd.eval})
    # Drop irrelevant columns and rename 'genres_v2'
    movie_genres.drop(
        columns=["Unnamed: 0", "genres", "movieId", "title"], inplace=True
    )
    movie_genres.rename(columns={"genres_v2": "genre_names"}, inplace=True)
    movie_genres["genre_values"] = movie_genres["genre_names"].apply(
        lambda row: map_genres(row, genres)
    )
    movie_genres["genre_count"] = movie_genres["genre_values"].apply(len)
    movie_genres.drop(columns=["genre_names"], inplace=True)
    # Create genre dictionary
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
    return genre_dict


def simulate_user(
    preferred_genres: list[int], genre_dict: dict[int, list[int]]
) -> tuple[list[int], list[float], str]:
    """Function used in A/B testing to simulate user history and grouping.

    Args:
        preferred_genres (list[int]): Genre ids that a user likes.
        genre_dict (dict[int, list[int]]): A dictionary where a genre is a key and the movie ids with that tag are items.

    Returns:
        tuple[list[int], list[float], str]: Movie ids a user enjoyed, their corresponding ratings and the user's group.
    """
    # Randomly sample U(10, 20) movies from the genres
    movie_ids_history = []
    for genre in preferred_genres:
        movie_ids_history.append(
            np.random.choice(
                genre_dict[genre], size=np.random.randint(low=5, high=11), replace=False
            ).tolist()
        )
    movie_ids_history = [movie for sublist in movie_ids_history for movie in sublist]
    # Randomly sample high ratings for said movies
    movie_ratings = np.random.choice(
        [7.0, 8.0, 9.0, 10.0], size=len(movie_ids_history)
    ).tolist()
    # Randomly assign the user to a group
    group = np.random.choice(["A", "B"])

    return movie_ids_history, movie_ratings, group


def find_user_trait_vector(
    u_mat: np.array,
    v_mat: np.array,
    bias_movie: np.array,
    lmd: float,
    tau: float,
    usr_history: list[int],
    usr_ratings: list[float],
) -> np.array:
    """Find the user trait vector for a new user with a set of previous ratings.

    Assume a user bias of zero.

    Args:
        u_mat (np.array): Trained user matrix, U.
        v_mat (np.array): Trained movie matrix, V.
        bias_movie (np.array): Trained movie biases, b_n^i.
        lmd (float): Objective function regulariser, lambda.
        tau (float): User matrix regulariser, tau.
        usr_history (list[int]): List of movie ids that a user rated.
        usr_ratings (list[float]): User's ratings of movies.

    Returns:
        np.array: User trait vector.
    """
    # Derive latent dimension
    latent_dim = u_mat.shape[1]
    # Compute user trait vector
    # Assume zero user bias
    ratings_term = 0
    vv_mat = 0
    # Iterate over movies rated by user
    for cnt, movie_id in enumerate(usr_history):
        ratings_term += (usr_ratings[cnt] - bias_movie[movie_id] - 0) * v_mat[
            movie_id, :
        ]
        vv_mat += np.matmul(
            v_mat[movie_id, :].reshape((latent_dim, 1)),
            v_mat[movie_id, :].reshape((1, latent_dim)),
        )
    # Use cholesky decomposition to find the user trait vector
    chol_factor, low = cho_factor(lmd * vv_mat + tau * np.identity(latent_dim))
    user_trait_vector = cho_solve(
        (chol_factor, low), lmd * ratings_term.reshape((latent_dim, 1))
    ).reshape((latent_dim,))
    return user_trait_vector


def top_n_recommendations(
    user_trait_vector: np.array,
    v_mat: np.array,
    bias_movie: np.array,
    movie_ids: pd.DataFrame,
    num_recom: int,
    remove_movies_limit: int,
    ratings: pd.DataFrame,
) -> pd.DataFrame:
    """Generate top n recommendations given a user trait vector.

    Args:
        user_trait_vector (np.array): Encoding of user preference.
        v_mat (np.array): Trained movie matrix, V.
        bias_movie (np.array): Trained movie biases, b_n^i.
        movie_ids (pd.DataFrame): Dataframe relating movie_ids and adapted ids.
        num_recom (int): Number of recommendations to return.
        remove_movies_limit (int): Exclude movies that fewer than remove_movies_limit users rated.
        ratings (pd.DataFrame): Dataframe of user-movie pairs with ratings.

    Returns:
        pd.DataFrame: Movie ids and corresponding scores.
    """
    number_of_movies = v_mat.shape[0]
    # Compute score for each movie
    score_for_movie = np.zeros((number_of_movies,))
    for movie in range(number_of_movies):
        score_for_movie[movie] = (
            np.dot(user_trait_vector, v_mat[movie, :]) + 0.05 * bias_movie[movie]
        )
    movie_scores = pd.DataFrame()
    movie_scores["Scores"] = score_for_movie
    movie_scores = movie_scores.reset_index()
    movie_scores.columns = ["movieId_order", "Scores"]
    movie_scores = pd.merge(movie_scores, movie_ids)
    movie_scores = movie_scores.sort_values(by="Scores", ascending=False)
    # Remove movies that were only rated by a few people
    movie_id_exclude = list(
        ratings["movieId_order"]
        .value_counts()
        .loc[lambda x: x < remove_movies_limit]
        .to_frame()
        .index
    )
    movie_scores_exclude = movie_scores[
        ~movie_scores["movieId_order"].isin(movie_id_exclude)
    ]

    return movie_scores_exclude.head(num_recom)
