import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedTk
from tkinter import filedialog
import random

def generate_random_data(n_users=50, n_items=50):
    users = [f'User{i}' for i in range(1, n_users + 1)]
    items = [f'Item{i}' for i in range(1, n_items + 1)]

    data = {
        'User': [],
        'Item': [],
        'Rating': []
    }

    for user in users:
        for item in items:
            if random.random() < 0.8:  # 80% chance of a user rating an item
                rating = random.randint(1, 5)  # Generate a random rating between 1 and 5
                data['User'].append(user)
                data['Item'].append(item)
                data['Rating'].append(rating)

    return data

data = generate_random_data()

df = pd.DataFrame(data)
user_item_matrix = df.pivot_table(index='User', columns='Item', values='Rating').fillna(0)
user_similarity_values = cosine_similarity(user_item_matrix)
user_similarity_values = np.clip(user_similarity_values, -1, 1)
user_similarity = pd.DataFrame(user_similarity_values, index=user_item_matrix.index, columns=user_item_matrix.index)


def get_recommendations(user, user_similarity, user_item_matrix, n_recommendations=2):
    if user not in user_similarity.index:
        return None

    weighted_ratings = (user_similarity.loc[user].values[:, np.newaxis] * user_item_matrix).sum(axis=0) / user_similarity.loc[user].abs().sum()
    weighted_ratings = pd.Series(weighted_ratings, index=user_item_matrix.columns)
    recommendations = weighted_ratings[user_item_matrix.loc[user] == 0].sort_values(ascending=False)

    return recommendations.head(n_recommendations)

# tkinter window

def show_recommendations():
    user = user_combobox.get()
    recommendations = get_recommendations(user, user_similarity, user_item_matrix)
    output.delete(1.0, tk.END)
    if recommendations is not None:
        output.insert(tk.END, f"Recommendations for {user}:\n")
        for item, score in recommendations.items():
            output.insert(tk.END, f"{item}: {score}\n")
    else:
        output.insert(tk.END, "User not found.")

def export_data_to_csv():
    df_to_export = pd.DataFrame(data)
    df_to_export.to_csv('data.csv', index=False)
    output.delete(1.0, tk.END)
    output.insert(tk.END, "Data exported to data.csv")

def load_data_from_csv():
    global data, df, user_item_matrix, user_similarity, user_list

    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        df = pd.read_csv(file_path)
        data = {'User': df['User'].tolist(), 'Item': df['Item'].tolist(), 'Rating': df['Rating'].tolist()}
        user_item_matrix = df.pivot_table(index='User', columns='Item', values='Rating').fillna(0)
        user_similarity = pd.DataFrame(cosine_similarity(user_item_matrix), index=user_item_matrix.index, columns=user_item_matrix.index)

        # Update user list in the drop-down
        user_list = df['User'].unique().tolist()
        user_combobox['values'] = user_list

        output.delete(1.0, tk.END)
        output.insert(tk.END, "Data loaded from CSV.\n")

root = ThemedTk(theme='radiance')
root.title("Collaborative Filtering Recommender System")

frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

user_label = ttk.Label(frame, text="Select User:")
user_label.grid(row=0, column=0, sticky=(tk.W), pady=(5, 0))

user_list = df['User'].unique().tolist()
user_combobox = ttk.Combobox(frame, values=user_list, state="readonly", width=18)
user_combobox.grid(row=0, column=1, pady=(5, 0))

get_recommendations_button = ttk.Button(frame, text="Get Recommendations", command=show_recommendations)
get_recommendations_button.grid(row=1, column=0, columnspan=2, pady=(10, 0))

output = tk.Text(frame, wrap=tk.WORD, width=40, height=10)
output.grid(row=2, column=0, columnspan=2, pady=(10, 0))

export_button = ttk.Button(frame, text="Export Data to CSV", command=export_data_to_csv)
export_button.grid(row=3, column=0, columnspan=2, pady=(10, 0))

load_button = ttk.Button(frame, text="Load Data from CSV", command=load_data_from_csv)
load_button.grid(row=3, column=4, columnspan=1, pady=(10, 0))

root.mainloop()
