"""Embedding results for prac 2."""
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE

from functions import create_ratings_df

# Get the ratings, as processed previously in prac 2
ratings = create_ratings_df("ratings_small.csv")
movie_ids = ratings[["movieId", "movieId_order"]].drop_duplicates()
# Merge with movies csv
movie_pd = pd.read_csv("movies_small.csv")
# Start movie ids at 0
movie_pd["movieId"] = movie_pd["movieId"] - 1
movie_pd = movie_pd[["movieId", "title"]]
movie_ids = pd.merge(movie_ids, movie_pd)
movie_ids = movie_ids.sort_values(by="movieId")
movie_ids.to_csv("implicit_feedback/movie_ids_small.csv")

# --- tSNE plots ---
with open("implicit_feedback/param/v_matrix_10.npy", "rb") as f:
    V_mat = np.load(f)

colours = []
movie_names = []
df = pd.DataFrame()
idx = 0
# Collect movie trait vectors in a dataframe
for movie in [1278, 2749, 3052, 3447, 5372, 5510, 5676]:
    df[f"LOTR_{idx}"] = V_mat[movie, :]
    df_mini = movie_ids.loc[movie_ids["movieId_order"] == movie]
    movie_names.append(df_mini["title"])
    colours.append("LOTR")
    idx += 1
idx = 0
for movie in [3775, 6280, 0, 1856, 4933, 3183, 2701, 4055, 4649, 5280, 5669]:
    df[f"KID_{idx}"] = V_mat[movie, :]
    df_mini = movie_ids.loc[movie_ids["movieId_order"] == movie]
    movie_names.append(df_mini["title"])
    colours.append("Children")
    idx += 1
idx = 0
for movie in [1183, 1608, 1830, 1077, 2982, 5462, 1184]:
    df[f"HORROR_{idx}"] = V_mat[movie, :]
    df_mini = movie_ids.loc[movie_ids["movieId_order"] == movie]
    movie_names.append(df_mini["title"])
    colours.append("Horror")
    idx += 1
idx = 0
for movie in [4575, 3106, 5132, 3486, 250, 267, 980, 2379]:
    df[f"ROMCOM_{idx}"] = V_mat[movie, :]
    df_mini = movie_ids.loc[movie_ids["movieId_order"] == movie]
    movie_names.append(df_mini["title"])
    colours.append("Rom Com")
    idx += 1
idx = 0
for movie in [2755, 4145, 2859, 4688, 4666]:
    df[f"WAR_{idx}"] = V_mat[movie, :]
    df_mini = movie_ids.loc[movie_ids["movieId_order"] == movie]
    movie_names.append(df_mini["title"])
    colours.append("War")
    idx += 1

# Reduce the encodings to two dimensions
tsne = TSNE(n_components=2, random_state=0, perplexity=12)
projections = tsne.fit_transform(df.T)

fig = px.scatter(
    projections,
    x=0,
    y=1,
    color=colours,
    labels={"color": "Genre"},
    hover_data=[movie_names],
    width=800,
    height=700,
)
fig.update_layout(
    title="tSNE Embeddings on movie matrix V",
    xaxis_title="Dimension 1",
    yaxis_title="Dimension 2",
)
fig.update_traces(
    marker={"size": 10, "line": {"width": 1.5, "color": "DarkSlateGrey"}},
    selector={"mode": "markers"},
)
fig.write_html("implicit_feedback/figures/tSNE.html")
fig.write_image("implicit_feedback/figures/tSNE.png")
fig.show()
