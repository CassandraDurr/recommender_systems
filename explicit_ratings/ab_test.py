"""Module that performs A/B testing."""
import random
import warnings
import pandas as pd
import numpy as np
from functions import (
    genre_key_dict,
    simulate_user,
    find_user_trait_vector,
    top_n_recommendations,
    create_ratings_df,
)

# Simulate d dummy users
dummyUsers = 200
# Find top n recommendations
numRecommendations = 20
# Log user_id, item_id, feedback on item_id, group (a or b)

# Mapping from genre name to integers
genres = {
    "Horror": 0,
    "War": 1,
    "Crime": 2,
    "IMAX": 3,
    "Western": 4,
    "Children": 5,
    "Adventure": 6,
    "(no genres listed)": 7,
    "Drama": 8,
    "Romance": 9,
    "Thriller": 10,
    "Film-Noir": 11,
    "Animation": 12,
    "Comedy": 13,
    "Sci-Fi": 14,
    "Fantasy": 15,
    "Action": 16,
    "Musical": 17,
    "Mystery": 18,
    "Documentary": 19,
}

# Pairs of genres we typically expect an individual to like
reasonable_genre_pairs = [
    [14, 15],  # Sci-Fi and Fantasty
    [12, 5],  # Animation and Children
    [15, 6],  # Fantasy and Adventure
    [11, 18],  # Film-noir and Mystery
    [11, 2],  # Film-noir and Crime
    [13, 9],  # Comedy and Romance
    [1, 16],  # War and Action
    [10, 0],  # Thriller and Horror
    [8, 9],  # Drama and Romance
    [5, 6],  # Children and Adventure
    [18, 2],  # Mystery and Crime
    [4, 16],  # Western and Action
]
genre_dict = genre_key_dict(file_location="movies_25m_genres_full.csv", genres=genres)

# Obtain U, V and movie_bias for the control and treatment group.
# Control group
with open("explicit_ratings/param/u_matrix_25.npy", "rb") as f:
    control_u = np.load(f)
with open("explicit_ratings/param/v_matrix_25.npy", "rb") as f:
    control_v = np.load(f)
with open("explicit_ratings/param/bias_movie_25.npy", "rb") as f:
    control_movie_bias = np.load(f)
# Treatment group
with open("explicit_ratings/param/u_matrix_30_genre.npy", "rb") as f:
    treat_u = np.load(f)
with open("explicit_ratings/param/v_matrix_30_genre.npy", "rb") as f:
    treat_v = np.load(f)
with open("explicit_ratings/param/bias_movie_30_genre.npy", "rb") as f:
    treat_movie_bias = np.load(f)

# Simulate users
usr_histories = []
usr_ratings = []
usr_groupings = []
for usr in range(dummyUsers):
    usr_simulation = simulate_user(
        random.choices(reasonable_genre_pairs)[0], genre_dict
    )
    usr_histories.append(usr_simulation[0])
    usr_ratings.append(usr_simulation[1])
    usr_groupings.append(usr_simulation[2])

# For each user, derive user trait vector and find recommendations
lmd = 0.1
tau = 0.01
# Dataframe linking movieId to movieId_order with titles
movie_ids = pd.read_csv("movie_ids.csv")
# Create ratings dataframe, as before
ratings = create_ratings_df(file_name="ratings_25m.csv")
# Store experiment results
logging = []
for usr in range(dummyUsers):
    # Determine group of user
    if usr_groupings[usr] == "A":
        # Control group
        # Find user trait vector
        user_trait_vector = find_user_trait_vector(
            control_u,
            control_v,
            control_movie_bias,
            lmd,
            tau,
            usr_histories[usr],
            usr_ratings[usr],
        )
        # Find top 20 recommendations
        recomm = top_n_recommendations(
            user_trait_vector,
            control_v,
            control_movie_bias,
            movie_ids,
            numRecommendations,
            90,
            ratings,
        )
        print(recomm)
    elif usr_groupings[usr] == "B":
        # Treatment group
        # Find user trait vector
        user_trait_vector = find_user_trait_vector(
            treat_u,
            treat_v,
            treat_movie_bias,
            lmd,
            tau,
            usr_histories[usr],
            usr_ratings[usr],
        )
        # Find top 20 recommendations
        recomm = top_n_recommendations(
            user_trait_vector,
            treat_v,
            treat_movie_bias,
            movie_ids,
            numRecommendations,
            90,
            ratings,
        )
        print(recomm)
    else:
        warnings.warn(f"Group must be A or B, received {usr_groupings[usr]}")
    # Log user id, item id, feedback, control/treatment
