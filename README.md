# Movie Recommender Systems

This repository contains code for building two recommendation engines using the MovieLens dataset. One is built using explicit ratings data and is trained with alternating least squares, while the other is a Bayesian Personalised Ranking (BPR) model built using curated implicit feedback data and is trained using with stochastic gradient descent. Both recommendation systems are built using matrix factorisation techniques which involves factorising the user-movie rating matrix to identify latent factors that capture user preferences and movie characteristics. Additionally, both models can incorporate genre information to improve the latent representations of the movie charcteristics. Trained recommender systems can be compared using the ``ab_test.py`` file. 

## Dataset
The dataset used for training and evaluating the recommendation models is the MovieLens dataset, which contains ratings for movies from a group of users. The model using explicit ratings data was trained using the MovieLens 25M dataset, whereas the model built on implicit feedback data was trained on the MovieLens 100K dataset. 
The dataset can be downloaded from the following link: https://grouplens.org/datasets/movielens/

## Getting Started
To use the code in this repository, you'll need to follow these steps:

1. Clone the repository onto your local machine using command: ``git clone https://github.com/CassandraDurr/recommender_systems.git``.
2. Install the required dependencies from ``requirements.txt`` using pip or conda.
3. GitHub does not permit the storage of large csv files in repositories. Four csv files were not pushed to the repository and must be downloaded to use the code as it is.
    - Download the 25M and 100K datasets from https://grouplens.org/datasets/movielens/.
    - Save the ratings and movie csvs from the 25M dataset as "ratings_25m.csv" and "movies_25m.csv" under the ``data`` folder.
    - Save the ratings and movie csvs from the 100K dataset as "ratings_small.csv" and "movies_small.csv" under the ``data`` folder.
3. Run the python files to train and evaluate the recommendation engines. Module docstrings are used to desribe the function of each python file. 

## Results
- Typical data used to train recommender engines exhibit power laws and this needs to be taken into account when making recommendations (e.g. down-weighting item biases when scoring films, filtering out movies infrequently rated and/ or accounting for variance with a Variational Bayes model).
- There are pros and cons to building a model with implicit feedback data vs explicit ratings data, and the best model will depend on its application and the data available, amongst other factors.
- The reduced dimension encodings from models trained on explicit ratings data appeared more reasonable than those from the model trained on implicit feedback data. However, the comparison is unfair due to the difference in dataset size used in training on account of computational constraints, and the fact that the implicit data was curated from explicit ratings.
- Increasing the complexity of a recommender system will not guarantee increased performance as seen in the controlled experiment. Increasing model complexity can also substantially increase computational burden in training as seen in the training of the implicit model with negative sampling based on genre information.

## Credits
This repository was created by Cassandra Durr, durrcass@gmail.com.



