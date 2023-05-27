import pandas as pd
from sklearn.ensemble import IsolationForest

# Load the dataset
df = pd.read_csv('ml-latest-small/ratings.csv')

# Extract the ratings column as a NumPy array
X = df['rating'].values.reshape(-1, 1)

# Create an instance of the IsolationForest class
isolation_forest = IsolationForest(n_estimators=100, contamination='auto')

# Fit the model to the data
isolation_forest.fit(X)

# Predict the anomaly scores for all data points
scores = isolation_forest.decision_function(X)

# Identify the anomalies as data points with negative anomaly scores
anomalies = X[scores < 0]

# Count the number of anomalies
num_anomalies = len(anomalies)
print("Number of anomalies:", num_anomalies)
