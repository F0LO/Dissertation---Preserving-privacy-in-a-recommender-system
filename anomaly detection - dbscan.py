import pandas as pd
from sklearn.cluster import DBSCAN

# Load the dataset
df = pd.read_csv('ml-latest-small/ratings.csv')

# Extract the ratings column as a NumPy array
X = df['rating'].values.reshape(-1, 1)

# Create an instance of the DBSCAN class
dbscan = DBSCAN(eps=0.25, min_samples=2)

# Fit the model to the data
dbscan.fit(X)

# Identify the anomalies as data points with cluster label -1
anomalies = X[dbscan.labels_ == -1]

# Count the number of anomalies
num_anomalies = len(anomalies)
print("Number of anomalies:", num_anomalies)
