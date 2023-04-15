# Movie Recommender Systems

This repository contains code for building two recommendation engines using the MovieLens dataset. One is built using explicit ratings data and is trained with alternating least squares, while the other is a Bayesian Personalised Ranking (BPR) model built using curated implicit feedback data and is trained using with stochastic gradient descent. Both recommendation systems are built using matrix factorisation techniques which involves factorising the user-movie rating matrix to identify latent factors that capture user preferences and movie characteristics. Additionally, both models can incorporate genre information to improve the latent representations of the movie charcteristics. 

## Getting Started
To use the code in this repository, you'll need to follow these steps:

1. Clone the repository onto your local machine.
2. Install the required dependencies using pip or conda.
3. Run the python files to train and evaluate the recommendation engines.

## Dependencies 
The following dependencies are required to run the code in this repository:
TODO

## Dataset
The dataset used for training and evaluating the recommendation models is the MovieLens dataset, which contains ratings for movies from a group of users. The model using explicit ratings data was trained using the MovieLens 25M dataset, whereas the model built on implicit feedback data was trained on the MovieLens 100K dataset. 
The dataset can be downloaded from the following link: https://grouplens.org/datasets/movielens/

## Results
TODO

## Credits
This repository was created by Cassandra Durr, durrcass@gmail.com.



