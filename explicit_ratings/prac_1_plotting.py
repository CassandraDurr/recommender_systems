"""Code to produce power plot and various data exploration histograms."""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns

ratings = pd.read_csv("data/ratings_25m.csv")
# use 1 to 10 scale to work in integers
ratings["rating_10"] = ratings["rating"] * 2

plotting = False
if plotting:
    value_cnts = (
        ratings["rating"]
        .value_counts()
        .reset_index()
        .sort_values(by="index")
        .reset_index(drop=True)
    )

    # Plots to understand the data
    millions_ticks_y = ticker.FuncFormatter(lambda x, pos: "{0:g}".format(x / 1e6))
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(value_cnts["index"], value_cnts["rating"], color="#1E63A4", width=0.5)
    ax.yaxis.set_major_formatter(millions_ticks_y)
    plt.xlabel("Ratings")
    plt.ylabel("Count (millions)")
    plt.title("Count per rating in millions")
    plt.savefig("figures/ratings_count.png")
    plt.show()

    # Mean rating per user
    mean_pu = ratings[["userId", "rating"]].groupby(["userId"]).mean()
    thousand_ticks_y = ticker.FuncFormatter(lambda x, pos: "{0:g}".format(x / 1e3))
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(mean_pu, color="#F05225", width=0.5)
    ax.yaxis.set_major_formatter(thousand_ticks_y)
    plt.xlabel("Mean rating per user")
    plt.ylabel("Frequency (thousands)")
    plt.title("Frequency of mean user ratings")
    plt.savefig("figures/ratings_meanpu.png")
    plt.show()

    # Mean rating per movie
    mean_pm = ratings[["movieId", "rating"]].groupby(["movieId"]).mean()
    thousand_ticks_y = ticker.FuncFormatter(lambda x, pos: "{0:g}".format(x / 1e3))
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(mean_pm, color="#FFB30F", width=0.5)
    ax.yaxis.set_major_formatter(thousand_ticks_y)
    plt.xlabel("Mean rating per movie")
    plt.ylabel("Frequency (thousands)")
    plt.title("Frequency of mean movie ratings")
    plt.savefig("figures/ratings_meanpm.png")
    plt.show()

    # Frequency vs rating per user
    mean_pu = ratings[["userId", "rating"]].groupby(["userId"]).median().reset_index()
    mean_pu.columns = ["userId", "medRating"]
    num_rating = (
        ratings["userId"]
        .value_counts()
        .reset_index()
        .sort_values(by="index")
        .reset_index(drop=True)
    )
    num_rating.columns = ["userId", "frequencyRating"]
    result = pd.merge(mean_pu, num_rating)

    plt.figure(figsize=(9, 6))
    sns.regplot(
        x=result["frequencyRating"],
        y=result["medRating"],
        scatter_kws={"color": "#1E63A4"},
        line_kws={"color": "red"},
    )
    plt.xscale("log")
    plt.xlabel("Log frequency of user rating")
    plt.ylabel("Median user rating")
    plt.ylim([0, 5.1])
    plt.title("Frequency of user ratings against the median rating per user")
    plt.savefig("figures/ratings_freq_vs_user_rating.png")
    plt.show()

plotPowerLaw = False
if plotPowerLaw:
    # Power laws investigation
    # Number of ratings per person = degree
    # Users
    num_rating = (
        ratings["userId"]
        .value_counts()
        .reset_index()
        .sort_values(by="index")
        .reset_index(drop=True)
    )
    num_rating.columns = ["userId", "degree"]
    freq = (
        num_rating["degree"]
        .value_counts()
        .reset_index()
        .sort_values(by="index")
        .reset_index(drop=True)
    )
    freq.columns = ["degree", "frequency"]
    # Movies
    num_rating_movie = (
        ratings["movieId"]
        .value_counts()
        .reset_index()
        .sort_values(by="index")
        .reset_index(drop=True)
    )
    num_rating_movie.columns = ["movieId", "degree"]
    freq_movie = (
        num_rating_movie["degree"]
        .value_counts()
        .reset_index()
        .sort_values(by="index")
        .reset_index(drop=True)
    )
    freq_movie.columns = ["degree", "frequency"]
    # Plot
    plt.figure(figsize=(9, 6))
    plt.scatter(
        x=freq_movie["degree"],
        y=freq_movie["frequency"],
        c="#FFB30F",
        label="movies",
        alpha=0.7,
    )
    plt.scatter(
        x=freq["degree"], y=freq["frequency"], c="#F05225", label="users", alpha=0.7
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Degree (log scale)")
    plt.ylabel("Frequency (log scale)")
    plt.legend()
    plt.savefig("figures/power_plot.png")
    plt.show()
