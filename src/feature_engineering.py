import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

print("Loading cleaned movie dataset...")

movies = pd.read_csv("data/processed/movies_cleaned.csv")

print("Preparing genre features...")

movies["genres"] = movies["genres"].fillna("")

# ✅ FIX: Keep comma-separated format consistent — used later for genre_overlap()
# (Original code replaced "," with " " which broke genre_overlap set splitting)
movies["genres"] = movies["genres"].str.replace(",", " ")

# ✅ FIX: Removed the duplicate genre trick ("genres + genres") — that was redundant.
#         Now using genres as the content directly for TF-IDF.
movies["content"] = movies["genres"]

print("Applying TF-IDF vectorization...")

# ✅ NEW: Added ngram_range and min_df for slightly richer genre features
tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)

tfidf_matrix = tfidf.fit_transform(movies["content"])

print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

# ✅ FIX: Ensure models directory exists
os.makedirs("models", exist_ok=True)

print("Saving TF-IDF matrix and vectorizer...")

with open("models/tfidf_matrix.pkl", "wb") as f:
    pickle.dump(tfidf_matrix, f)

# ✅ NEW: Save the vectorizer too — useful if you want to transform new inputs later
with open("models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("Saving movie metadata...")

movies.to_pickle("models/movies_metadata.pkl")

print("Feature engineering completed.")