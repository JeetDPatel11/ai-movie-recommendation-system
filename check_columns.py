import pandas as pd

movies = pd.read_pickle("models/movies_metadata.pkl")

print("=== Columns ===")
print(movies.columns.tolist())

print("\n=== Shape ===")
print(movies.shape)

print("\n=== Sample Rows ===")
print(movies.head(3).to_string())

print("\n=== Null Counts ===")
print(movies.isnull().sum())

print("\n=== Dtypes ===")
print(movies.dtypes)