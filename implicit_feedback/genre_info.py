"""Find the genres of the movies for prac 2, using the 10K dataset."""
import pandas as pd

# Incorporating genre information
# ratings = pd.read_csv("ratings_small.csv")
ratings = pd.read_csv("ratings_25m.csv")
ratings = ratings[ratings["rating"]>=4]
ratings = ratings.drop(columns="timestamp")
# use 1 to 10 scale to work in integers
ratings["rating_10"] = ratings["rating"] * 2
# Start the movie ratings at 0
ratings["movieId_lessone"] = ratings["movieId"] - 1
# Shift movie ids so that ids go from 0 to max 
idShift = pd.DataFrame()
idShift["movieId_lessone"] = ratings["movieId_lessone"].unique().copy()
idShift = idShift.sort_values(by="movieId_lessone")
idShift.reset_index(drop=True, inplace=True)
idShift.reset_index(drop=False, inplace=True)
idShift.columns = ["movieId_order", "movieId_lessone"]
# Combine dataframes on "movieId_lessone"
ratings = pd.merge(ratings, idShift)

movie_ids = ratings[["movieId","movieId_order"]].drop_duplicates()

# movie_pd = pd.read_csv("movies_small.csv")
movie_pd = pd.read_csv("movies_25m.csv")
movie_pd = movie_pd[["movieId", "title", "genres"]]
movie_ids = pd.merge(movie_ids, movie_pd)
movie_ids = movie_ids.sort_values(by="movieId_order")
movie_ids["genres_v2"] = movie_ids["genres"].str.split("|")
movie_ids.reset_index(inplace=True, drop=True)

print(movie_ids.head(20))

# movie_ids.to_csv("movies_small_genres.csv")
movie_ids.to_csv("movies_25m_genres.csv")
