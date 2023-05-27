import pandas as pd
import numpy as np
# Load data
ratings = pd.read_csv('ml-latest-small/ratings.csv')
# Set privacy parameter
epsilon = 0.1
# Calculate sensitivity
sensitivity = 1 / epsilon
# Add noise to ratings
ratings['rating'] += np.random.laplace(loc=0, scale=sensitivity, size=len(ratings))
# Save anonymized data
ratings.to_csv('ratings_anonymized.csv', index=False)
