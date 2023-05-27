import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

# Load the dataset
df = pd.read_csv('ml-latest-small/ratings.csv')

# Extract the ratings column as a NumPy array
X = df['rating'].values.reshape(-1, 1)

# Create an instance of the LocalOutlierFactor class
lof = LocalOutlierFactor(n_neighbors=20, contamination='auto')

# Fit the model to the data and predict the anomaly scores
scores = lof.fit_predict(X)

# Identify the anomalies as data points with negative anomaly scores
anomalies = X[scores < 0]

# Count the number of anomalies
num_anomalies = len(anomalies)
print("Number of anomalies:", num_anomalies)
