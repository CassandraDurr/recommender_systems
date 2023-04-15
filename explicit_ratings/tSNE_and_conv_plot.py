"""Plots for loglikelihood, RMSE and tSNE (reduce dimensions of movie matrix V)."""
from matplotlib import ticker
from sklearn.manifold import TSNE
import plotly.express as px
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Convergence ---
with open("explicit_ratings/param/loglik_genre.npy", "rb") as f:
    loglik = np.load(f)

with open("explicit_ratings/param/rmse_vals_genre.npy", "rb") as f:
    rmse_vals = np.load(f)


# Plot log likelihood
millions_ticks_y = ticker.FuncFormatter(lambda x, pos: "{0:g}".format(x / 1e6))
fig, ax = plt.subplots(figsize=(9, 6))
ax.yaxis.set_major_formatter(millions_ticks_y)
plt.plot(loglik, color="#1E63A4")
# plt.axhline(
#     loglik[24],
#     color="#FFB30F",
#     linestyle="dotted",
#     label=f"LL at iteration 25, {loglik[24]}",
# )
plt.axhline(
    loglik[-1], color="#F05225", linestyle="dotted", label=f"Final LL, {loglik[-1]}"
)
plt.xlabel("Iteration")
plt.ylabel("Log likelihood (millions)")
plt.title("Log likelihood over training iterations")
plt.legend()
plt.savefig("explicit_ratings/figures/loglikelihood_genre.png")
plt.show()

# Plot RMSE
plt.figure(figsize=(9, 6))
plt.plot(rmse_vals, color="#1E63A4")
# plt.axhline(
#     rmse_vals[24],
#     color="#FFB30F",
#     linestyle="dotted",
#     label=f"RMSE at iteration 25, {rmse_vals[24]}",
# )
plt.axhline(
    rmse_vals[-1],
    color="#F05225",
    linestyle="dotted",
    label=f"Final RMSE, {rmse_vals[-1]}",
)
plt.xlabel("Iteration")
plt.ylabel("RMSE")
plt.title("RMSE over training iterations")
plt.legend()
plt.savefig("explicit_ratings/figures/RMSE_genre.png")
plt.show()

# --- tSNE plots ---
with open("explicit_ratings/param/v_matrix_30_genre.npy", "rb") as f:
    V_mat = np.load(f)


movie_ids = pd.read_csv("movie_ids.csv")
colours = []
movie_names = []
df = pd.DataFrame()
idx = 0
# Collect movie trait vectors in a dataframe
for movie in [2026, 4887, 5840, 7028, 18913, 20531, 23320]:
    df[f"LOTR_{idx}"] = V_mat[movie, :]
    df_mini = movie_ids.loc[movie_ids["movieId_order"] == movie]
    movie_names.append(df_mini["title"])
    colours.append("LOTR")
    idx += 1
idx = 0
for movie in [0, 3021, 14803, 56473, 4780, 6258, 8246, 50315, 9969]:
    df[f"KID_{idx}"] = V_mat[movie, :]
    df_mini = movie_ids.loc[movie_ids["movieId_order"] == movie]
    movie_names.append(df_mini["title"])
    colours.append("Children")
    idx += 1
idx = 0
for movie in [2903, 2988, 1683, 2607, 3436, 34994, 1885]:
    df[f"HORROR_{idx}"] = V_mat[movie, :]
    df_mini = movie_ids.loc[movie_ids["movieId_order"] == movie]
    movie_names.append(df_mini["title"])
    colours.append("Horror")
    idx += 1
idx = 0
for movie in [15593, 7168, 13555, 12333, 6043, 16706]:
    df[f"ROMCOM_{idx}"] = V_mat[movie, :]
    df_mini = movie_ids.loc[movie_ids["movieId_order"] == movie]
    movie_names.append(df_mini["title"])
    colours.append("Rom Com")
    idx += 1
idx = 0
for movie in [5208, 13440, 10372, 4904, 12825, 12926]:
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
fig.write_html("explicit_ratings/figures/tSNE_genre.html")
fig.write_image("explicit_ratings/figures/tSNE_genre.png")
fig.show()
