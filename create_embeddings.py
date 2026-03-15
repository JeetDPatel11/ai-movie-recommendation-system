import pandas as pd
import pickle
import os
from sentence_transformers import SentenceTransformer  # type: ignore

print("Loading dataset...")

movies = pd.read_pickle("models/movies_metadata.pkl")

print(f"Total movies: {len(movies)}")

print("Preparing text for embeddings...")

# ✅ FIX: Restore commas in genres for readability in the sentence (was replaced with spaces in feature_engineering)
genres_text = movies["genres"].astype(str).str.replace(" ", ", ")

# ✅ IMPROVEMENT: Richer combined text — title + year + genres
#    Original only used title + genres + content (content was just genres again — redundant)
movies["combined_text"] = (
    movies["title"].astype(str) + ". " +
    "Released in " + movies["year"].astype(str) + ". " +
    "Genres: " + genres_text + "."
)

print("Sample combined text:")
print(movies["combined_text"].iloc[0])

print("\nLoading AI model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Creating semantic embeddings (this may take a while for large datasets)...")

# ✅ NEW: batch_size for memory efficiency on large datasets
embeddings = model.encode(
    movies["combined_text"].fillna("").tolist(),
    show_progress_bar=True,
    batch_size=64
)

print(f"Embeddings shape: {embeddings.shape}")

# ✅ FIX: Ensure models directory exists
os.makedirs("models", exist_ok=True)

print("Saving embeddings...")

with open("models/semantic_embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

print("Embeddings saved successfully!")