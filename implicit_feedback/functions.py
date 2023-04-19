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
        # User/ movie indices and star, end arrays per user
        self.user_idx = user_idx
        self.user_start_index = user_start_index
        self.user_end_index = user_end_index

    def predict(self, user: np.array, movie: np.array):
        """Prediction formula for BPR.

        Dot product between user trait vector and movie trait vector.

        Args:
            user (np.array): userId.
            movie (np.array): movieId.

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

    def sample_negative_per_user_naive(
        self, user: int, movie_frequencies: np.array
    ) -> int:
        """Sample a negative movie per user.

        Args:
            user (int): userId.
            movie_frequencies (np.array): Array of probabilities of each movie occurring.

        Returns:
            int: Movie id which hasn't been watched by the user.
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
        return np.random.choice(neg_options, p=movie_frequencies, size=1)[0]

    def sample_negative_per_user_genre(
        self,
        user: int,
        pos_movie: int,
        movie_frequencies: np.array,
        genre_info: pd.DataFrame,
    ) -> int:
        """Sample a negative movie per user using genre information.

        Args:
            user (int): userId.
            pos_movie
            movie_frequencies (np.array): Array of probabilities of each movie occurring.
            genre_info (pd.DataFrame): Dataframe with movie ids [genres_v2] and their corresponding genres [genres_v2].

        Returns:
            int: Movie id which hasn't been watched by the user.
        """
        # u = user, i = positive movie, j = negative movie
        user_subset = self.user_idx[
            self.user_start_index[user] : self.user_end_index[user]
        ]

        # Positive movie genres
        pos_genres = genre_info.loc[
            genre_info["movieId_order"] == pos_movie, "genres_v2"
        ].values[0]

        # Acceptable negtives - naive
        neg_options = sorted(
            set(np.arange(self.num_items)).difference(set(user_subset[:, 1]))
        )

        # Probabilities corresponding to negatives
        movie_frequencies = self.force_probability_sum(movie_frequencies[neg_options])

        check_appropriate_choice = True
        while check_appropriate_choice:
            guess_random_choice = np.random.choice(
                neg_options, p=movie_frequencies, size=1
            )[0]
            # Check if random choice genre not in list of genres
            random_choice_genre = genre_info.loc[
                genre_info["movieId_order"] == guess_random_choice, "genres_v2"
            ].values[0]
            if len(set(random_choice_genre).intersection(set(pos_genres))) == 0:
                check_appropriate_choice = False
        return guess_random_choice

    def draw_triplet_per_user_naive(
        self, user: int, movie_frequencies: np.array
    ) -> np.array:
        """Naive sampling strategy to obtain a positive and negative movie sample per user.

        The strategy is naive because it doesn't account for movie genres.

        Args:
            user (int): userId
            movie_frequencies (np.array): Array of probabilities of each movie occurring.

        Returns:
            np.array: Array with [u, i, j]
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
        # Resultant u, i, j sample
        triplet = np.array(
            [
                user,
                np.random.choice(user_subset[:, 1]),
                np.random.choice(
                    neg_options, p=movie_frequencies, size=1, replace=False
                )[0],
            ]
        )
        return triplet

    def compute_gradients(
        self, triplet: np.array, x_uij: float
    ) -> tuple[np.array, np.array, np.array]:
        """Return the gradients of the user vector, positive movie vector and negative movie vector.

        Args:
            triplet (np.array): [u, i, j].
            x_uij (float): x_ui-x_uj.

        Returns:
            tuple[np.array, np.array, np.array]: Three gradients for SGD update.
        """
        co_efficient = np.exp(-x_uij) / (1 + np.exp(-x_uij))
        user_grad = co_efficient * (
            self.movie_matrix_v[triplet[1], :] - self.movie_matrix_v[triplet[2], :]
        )
        pos_matrix_grad = co_efficient * self.user_matrix_u[triplet[0], :]
        neg_matrix_grad = -1 * pos_matrix_grad
        return user_grad, pos_matrix_grad, neg_matrix_grad

    def sgd_update(
        self,
        triplet: np.array,
        gradients: tuple[np.array, np.array, np.array],
        regulariser: float,
    ) -> None:
        """Perform SGD update.

        Args:
            triplet (np.array): [u,i,j]
            gradients (tuple[np.array, np.array, np.array]): Output of compute_gradients function.
            regulariser (float): Regularisation value.
        """
        self.user_matrix_u[triplet[0], :] += self.learning_rate * (
            gradients[0] + regulariser * self.user_matrix_u[triplet[0], :]
        )
        self.movie_matrix_v[triplet[1], :] += self.learning_rate * (
            gradients[1] + regulariser * self.movie_matrix_v[triplet[1], :]
        )
        self.movie_matrix_v[triplet[2], :] += self.learning_rate * (
            gradients[2] + regulariser * self.movie_matrix_v[triplet[2], :]
        )

    def precision_and_recall_at_k(self, k: int) -> tuple[float, float]:
        """Find the mean precision at k and mean recall at k over all users.

        Args:
            k (int): Number of recommendations to make per user.

        Returns:
            tuple[float, float]: mean precision at k and mean recall at k.
        """
        # Store the precision and recall per user
        precision_at_k = []
        recall_at_k = []
        # Iterate over all users
        for user in range(self.num_users):
            user_subset = self.user_idx[
                self.user_start_index[user] : self.user_end_index[user]
            ]
            # Define the positive options for this user
            pos_options = sorted(set(user_subset[:, 1]))
            # Predict the scores for all items
            predicted_scores = []
            for movie in range(self.num_items):
                predicted_scores.append([movie, self.predict(user, movie)])
            # Sort predicted scores by score in descending order
            predicted_scores.sort(key=lambda x: x[1], reverse=True)
            # Find the top k movies by score
            top_k_movies = [movie for movie, score in predicted_scores[:k]]
            # How many of the top k movies are from the positive options?
            true_positive = len(set(top_k_movies).intersection(pos_options))
            precision_at_k.append(true_positive / k)
            recall_at_k.append(true_positive / len(pos_options))
        # Find the average precision at k and the average recall at k
        avg_precision_at_k = np.mean(precision_at_k)
        avg_recall_at_k = np.mean(recall_at_k)
        return avg_precision_at_k, avg_recall_at_k

    # Define a function to compute precision and recall at k for parallelisation
    def compute_precision_and_recall_at_k(self, k: int) -> tuple[float, float]:
        """Function for precision and recall at k for the purpose of parallelisation.

        Args:
            k (int): Number of recommendations to make per user.

        Returns:
            tuple[float, float]: mean precision at k and mean recall at k.
        """
        avg_precision_at_k, avg_recall_at_k = self.precision_and_recall_at_k(k=k)
        print(f"k={k}: Precision = {avg_precision_at_k}, Recall = {avg_recall_at_k}")
        return avg_precision_at_k, avg_recall_at_k


def create_ratings_df(file_name: str) -> pd.DataFrame:
    """Ingest a ratings csv with columns userId, movieId and ratings and adapt indices.

    Remove ratings lower than 4, or 8 in 10 star rating.

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
    # Drop ratings lower than 4 stars.
    print(f"Number of ratings using all stars: {ratings.shape[0]}")
    ratings = ratings[ratings["rating"] >= 4]
    print(f"Number of ratings using 4+ stars: {ratings.shape[0]}")
    # Drop users with less than 10 ratings given the above condition
    user_removal = ratings["userId"].value_counts().reset_index()
    user_removal = user_removal[user_removal["userId"] < 10]["index"].values
    ratings = ratings[~ratings["userId"].isin(user_removal)]
    print(f"Number of ratings with reduced number of users: {ratings.shape[0]}")
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
    id_shift = pd.DataFrame()
    id_shift["movieId"] = ratings["movieId"].unique().copy()
    id_shift = id_shift.sort_values(by="movieId")
    id_shift.reset_index(drop=True, inplace=True)
    id_shift.reset_index(drop=False, inplace=True)
    id_shift.columns = ["movieId_order", "movieId"]
    # Combine dataframes on "movieId"
    ratings = pd.merge(ratings, id_shift)
    # There is also an issue that not all users are present in the ratings data after dropping rows.
    id_shift = pd.DataFrame()
    id_shift["userId"] = ratings["userId"].unique().copy()
    id_shift = id_shift.sort_values(by="userId")
    id_shift.reset_index(drop=True, inplace=True)
    id_shift.reset_index(drop=False, inplace=True)
    id_shift.columns = ["userId_order", "userId"]
    # Combine dataframes on "userId"
    ratings = pd.merge(ratings, id_shift)
    print(
        f"User id order: min={np.min(ratings['userId_order'])}, max = {np.max(ratings['userId_order'])}, total = {ratings['userId_order'].nunique()}"
    )
    print(
        f"Movie id order: min={np.min(ratings['movieId_order'])}, max = {np.max(ratings['movieId_order'])}, total = {ratings['movieId_order'].nunique()}"
    )

    return ratings


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
        ratings_df (pd.DataFrame): Dataframe containing user & item ids.
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
