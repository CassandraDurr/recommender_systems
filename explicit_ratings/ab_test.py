"""Module that performs A/B testing."""
import random
import sys
import warnings
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from functions import (
    genre_key_dict,
    simulate_user,
    find_user_bias,
    find_user_trait_vector,
    top_n_recommendations,
    create_ratings_df,
)

# If performing an A/B test: True
performTest = False
# If evaluating an A/B test: True
evaluateTest = True

# If performing the test
if performTest:
    # Simulate d dummy users
    dummyUsers = 500
    # Find top n recommendations
    numRecommendations = 20

    # Mapping from genre name to integers/ ids
    genre_ids = {
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
        [13, 9],  # Comedy and Romance
        [1, 16],  # War and Action
        [10, 0],  # Thriller and Horror
        [8, 9],  # Drama and Romance
        [5, 6],  # Children and Adventure
        [18, 2],  # Mystery and Crime
        [4, 16],  # Western and Action
    ]
    genre_dict = genre_key_dict(
        file_location="movies_25m_genres_full.csv", genres=genre_ids
    )

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
    genre_pairs = []
    for usr in range(dummyUsers):
        genre_pairs.append(random.choices(reasonable_genre_pairs)[0])
        usr_simulation = simulate_user(genre_pairs[-1], genre_dict)
        usr_histories.append(usr_simulation[0])
        usr_ratings.append(usr_simulation[1])
        usr_groupings.append(usr_simulation[2])

    # For each user, derive user trait vector and find recommendations
    lmd = 0.1
    tau = 0.01
    alpha = 0.01
    # Dataframe linking movieId to movieId_order with titles
    movie_ids = pd.read_csv("movie_ids.csv")
    movie_ids.drop(columns=["Unnamed: 0"], inplace=True)
    # Create ratings dataframe, as before
    ratings = create_ratings_df(file_name="ratings_25m.csv")
    # Store experiment results
    logging = []
    for usr in range(dummyUsers):
        # Determine group of user
        if usr_groupings[usr] == "A":
            # Control group
            # Find user bias
            user_bias = find_user_bias(
                control_movie_bias,
                lmd,
                alpha,
                usr_histories[usr],
                usr_ratings[usr],
            )
            # Find user trait vector
            user_trait_vector = find_user_trait_vector(
                control_u,
                control_v,
                control_movie_bias,
                lmd,
                tau,
                usr_histories[usr],
                usr_ratings[usr],
                user_bias,
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
            # Log user id, item id, feedback per recommendation, control/treatment etc
            for _, row in recomm.iterrows():
                logging.append(
                    [
                        usr,
                        genre_pairs[usr][0],
                        genre_pairs[usr][1],
                        row["movieId_order"],
                        row["Scores"],
                        row["title"],
                        "A",
                    ]
                )
        elif usr_groupings[usr] == "B":
            # Treatment group
            # Find user bias
            user_bias = find_user_bias(
                treat_movie_bias,
                lmd,
                alpha,
                usr_histories[usr],
                usr_ratings[usr],
            )
            # Find user trait vector
            user_trait_vector = find_user_trait_vector(
                treat_u,
                treat_v,
                treat_movie_bias,
                lmd,
                tau,
                usr_histories[usr],
                usr_ratings[usr],
                user_bias,
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
            # Log user id, item id, feedback per recommendation, control/treatment etc
            for _, row in recomm.iterrows():
                logging.append(
                    [
                        usr,
                        genre_pairs[usr][0],
                        genre_pairs[usr][1],
                        row["movieId"],
                        row["Scores"],
                        row["title"],
                        "B",
                    ]
                )
        else:
            warnings.warn(f"group must be A or B, received {usr_groupings[usr]}")
            sys.exit()

    # Record logging list as a csv
    logging = pd.DataFrame(
        logging,
        columns=[
            "user_id",
            "genre_1",
            "genre_2",
            "movie_id",
            "score",
            "movie_title",
            "group",
        ],
    )
    # Link genre id to genre name by reversing the dictionary with genre and ids.
    reverse_genre_ids = {value: key for key, value in genre_ids.items()}
    logging["genre_1_name"] = [reverse_genre_ids[id] for id in logging["genre_1"]]
    logging["genre_2_name"] = [reverse_genre_ids[id] for id in logging["genre_2"]]
    # Change column order and omit genre ids
    logging = logging[
        [
            "user_id",
            "genre_1_name",
            "genre_2_name",
            "movie_id",
            "movie_title",
            "score",
            "group",
        ]
    ]
    # Store logging
    logging.to_csv("explicit_ratings/logging/AB_test_with_bias.csv", index=False)

if evaluateTest:
    logging = pd.read_csv("explicit_ratings/logging/AB_test_with_bias.csv")
    # Check the number of users in group A vs group B
    # A paired t-test cannot be performed if group sizes are unequal.
    group_cnts = logging["group"].value_counts() / 20
    group_cnts = group_cnts.reset_index()
    group_cnts.columns = ["group", "count"]
    # There should only be two groups - A and B.
    if group_cnts.shape[0] != 2:
        warnings.warn("The number of groupings exceeds 2.")
        sys.exit()
    if group_cnts["count"][0] != group_cnts["count"][1]:
        print("The number of users in control and treatment are not equal.")
        grp_a = group_cnts.loc[group_cnts["group"] == "A", "count"].values[0]
        grp_b = group_cnts.loc[group_cnts["group"] == "B", "count"].values[0]
        # Equalise the number of users in each group if there is a mismatch
        if grp_a > grp_b:
            # How many more A is there than B
            difference = grp_a - grp_b
            # User ids of A
            usr_ids = logging.loc[logging["group"] == "A", "user_id"].unique()
        if grp_a < grp_b:
            # How many more B is there than A
            difference = grp_b - grp_a
            # User ids of A
            usr_ids = logging.loc[logging["group"] == "B", "user_id"].unique()
        # Sample 'difference' user ids belong to A
        usr_ids = np.random.choice(usr_ids, size=int(difference))
        # Remove users from dataframe
        logging = logging[~logging["user_id"].isin(usr_ids)]

    # Split the groups into control and treatment
    control_group = logging[logging["group"] == "A"]
    treatment_group = logging[logging["group"] == "B"]

    # Check if populations have equal variances using Levene's test.
    # The t-test assumes equal population variances
    l_statistic, p_value = stats.levene(
        control_group["score"], treatment_group["score"]
    )
    print(f"Levene test result: \nt = {l_statistic}  p-value = {p_value}")

    if p_value < 0.05:
        print("Populations do not have equal variances.")
        print("Perform Welchs t-test if samples are normally distributed.")
        equalVariance = False
    else:
        print("Populations have equal variances.")
        print("Perform Welchs t-test if samples are normally distributed.")
        equalVariance = True

    # Use Shapiro Wilk to determine if samples are normally distributed.
    s_statistic_c, p_value_c = stats.shapiro(control_group["score"])
    print(
        f"Shapiro Wilk test control group: \ns = {s_statistic_c}  p-value = {p_value_c}"
    )
    s_statistic_t, p_value_t = stats.shapiro(treatment_group["score"])
    print(
        f"Shapiro Wilk test treatment group: \ns = {s_statistic_t}  p-value = {p_value_t}"
    )
    if (p_value_c > 0.05) and (p_value_t > 0.05):
        print("Both populations are likely Gaussian.")
        bothGaussian = True
    else:
        print("Both populations are not likely Gaussian.")
        print("Use the Wilcoxon rank-sum test instead.")
        bothGaussian = False

    # T-test to measure difference in top 20 scores, assuming normality.
    if bothGaussian:
        # 1. Two-tailed t-test (means are the same)
        t_statistic, p_value = stats.ttest_ind(
            control_group["score"], treatment_group["score"], equal_var=equalVariance
        )
        print(f"Two-tailed t-test result: \nt = {t_statistic}  p-value = {p_value}")
        if p_value < 0.05:
            # Reject the null hypothesis.
            print(
                "There is a significant difference between the means of the two groups."
            )
        else:
            print(
                "There is no significant difference between the means of the two groups."
            )

        # 2. One tailed t-test: control is better
        t_statistic, p_value = stats.ttest_ind(
            control_group["score"],
            treatment_group["score"],
            alternative="greater",
            equal_var=equalVariance,
        )
        print(
            f"One-tailed t-test result - A>B: \nt = {t_statistic}  p-value = {p_value}"
        )
        if p_value < 0.05:
            # Reject the null hypothesis.
            print("Control has a significantly higher mean than Treatment.")

        # 2. One tailed t-test: treatment is better
        t_statistic, p_value = stats.ttest_ind(
            treatment_group["score"],
            control_group["score"],
            alternative="greater",
            equal_var=equalVariance,
        )
        print(
            f"One-tailed t-test result - B>A: \nt = {t_statistic}  p-value = {p_value}"
        )
        if p_value < 0.05:
            # Reject the null hypothesis.
            print("Treatment has a significantly higher mean than Control.")
    else:
        # Wilcoxon rank-sum test
        # 1. Two-tailed test (medians are the same)
        s_statistic, p_value = stats.ranksums(
            control_group["score"], treatment_group["score"]
        )
        print(
            f"Two-tailed Wilcoxon rank-sum test result: \ns = {s_statistic}  p-value = {p_value}"
        )
        if p_value < 0.05:
            # Reject the null hypothesis.
            print(
                "There is a significant difference between the medians of the two groups."
            )
        else:
            print(
                "There is no significant difference between the medians of the two groups."
            )

        # 2. One tailed test: control is better
        s_statistic, p_value = stats.ranksums(
            control_group["score"],
            treatment_group["score"],
            alternative="greater",
        )
        print(
            f"Wilcoxon rank-sum test result - A>B: \nt = {s_statistic}  p-value = {p_value}"
        )
        if p_value < 0.05:
            # Reject the null hypothesis.
            print("Control has a significantly higher median than Treatment.")

        # 2. One tailed test: treatment is better
        s_statistic, p_value = stats.ranksums(
            treatment_group["score"],
            control_group["score"],
            alternative="greater",
        )
        print(
            f"Wilcoxon rank-sum test result - B>A: \nt = {s_statistic}  p-value = {p_value}"
        )
        if p_value < 0.05:
            # Reject the null hypothesis.
            print("Treatment has a significantly higher median than Control.")

    # Plot the scores for the control and treatment group
    plt.figure(figsize=(9, 5), constrained_layout=True)
    plt.boxplot(
        [control_group["score"], treatment_group["score"]],
        vert=False,
        patch_artist=True,
        labels=["Control", "Treatment"],
        # Outline
        boxprops={"linewidth": 1, "color": "#00235B"},
        # Whiskers
        whiskerprops={"linewidth": 1, "color": "#00235B"},
        # Outlier points
        flierprops={
            "marker": "o",
            "markerfacecolor": "#83C7AB",
            "markersize": 5,
            "alpha": 0.5,
        },
        # Median line
        medianprops={"linewidth": 1, "color": "#00235B"},
    )
    plt.title("Box plot of scores from both groups")
    # plt.gca().axes.get_xaxis().set_visible(False)
    plt.savefig("explicit_ratings/figures/boxplot_with_bias.png")
    plt.show()
