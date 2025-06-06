# SVD-Based Movie Recommender System

This project implements a **Collaborative Filtering** movie recommender system using **Singular Value Decomposition (SVD)** to predict user ratings and recommend movies effectively.

## Overview

Our model tackles the challenges of **data sparsity** and the abundance of **unrated items** in large-scale movie datasets by leveraging latent factor models. The key idea is to learn underlying patterns in user preferences and movie attributes by decomposing the user-item interaction matrix.

## Core Methodology

- **Matrix Decomposition**: The user-movie interaction matrix is decomposed into three matrices:
  - **U**: Represents users in latent space.
  - **S**: Diagonal matrix of singular values capturing concept strength.
  - **Vᵀ**: Encodes movies in terms of latent features.

- **Dimensionality Reduction**: To reduce computational complexity and noise, we truncate the decomposition to a lower-dimensional space (`k`), preserving only the most significant latent factors.

- **Similarity Computation**: 
  - Cosine similarity is calculated in the reduced latent space to evaluate the closeness between users and items.
  - This similarity enhances prediction accuracy by focusing on the most meaningful patterns.

- **Rating Prediction**: 
  - We use a **k-Nearest Neighbors (k-NN)** strategy.
  - For each user, the top `k` similar neighbors are selected.
  - Predicted ratings are computed as a **weighted sum** of the neighbors’ ratings.

## Evaluation

The model’s performance is measured using the following metrics:

- **Root Mean Squared Error (RMSE): 0.87**
- **Mean Absolute Error (MAE): 0.67**

Optimal values for `k` (number of neighbors and latent factors) were determined through extensive testing and validation.

---

This SVD-based recommender system offers a scalable and interpretable approach to generating personalized movie recommendations.

