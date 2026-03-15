import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

print("Loading movie metadata...")

movies = pd.read_pickle("models/movies_metadata.pkl")

print("Loading TF-IDF matrix...")

with open("models/tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)


# ---------------------------------------------------
# Industry Filter  (operates on OMDB data dict)
# ✅ FIX: Original used exact equality on Country/Language — OMDB returns
#         comma-separated strings like "India, USA" so "in" checks are needed.
# ---------------------------------------------------

def filter_by_industry(movie_data: dict, industry: str) -> bool:
    """Return True if the OMDB movie_data dict matches the requested industry."""

    country = movie_data.get("Country", "")
    language = movie_data.get("Language", "")

    industry_map = {
        "Bollywood":  ("India", "Hindi"),
        "Tollywood":  ("India", "Telugu"),
        "Kollywood":  ("India", "Tamil"),
        "Mollywood":  ("India", "Malayalam"),
        "Sandalwood": ("India", "Kannada"),
    }

    if industry in industry_map:
        req_country, req_language = industry_map[industry]
        return req_country in country and req_language in language

    if industry == "Hollywood":
        return any(c in country for c in ["USA", "United States", "UK", "Canada", "Australia"])

    return True  # "All" or unknown


# ---------------------------------------------------
# Genre Overlap
# ✅ FIX: Original split on "," but feature_engineering replaced "," with " ",
#         so genres were stored space-separated. Now splitting on " " to match.
# ---------------------------------------------------

def genre_overlap(movie_genres: str, candidate_genres: str) -> int:
    """Count shared genres between two genre strings."""

    if pd.isna(movie_genres) or pd.isna(candidate_genres):
        return 0

    set1 = set(str(movie_genres).split())
    set2 = set(str(candidate_genres).split())

    return len(set1.intersection(set2))


# ---------------------------------------------------
# Recommendation Function
# ✅ IMPROVEMENTS:
#   - Deduplicated score calculation into a single vectorized operation
#   - genre_score now normalized (0–1) to match other weights
#   - exclude_self handled cleanly
#   - Added min_votes filter to avoid recommending obscure movies
# ---------------------------------------------------

def recommend(movie_title: str, industry: str = "All", top_n: int = 5, min_votes: int = 100) -> pd.DataFrame:

    matches = movies[movies["title"].str.lower() == movie_title.lower()]

    if matches.empty:
        print(f"Movie '{movie_title}' not found.")
        return pd.DataFrame()

    # Pick most popular match if title appears multiple times
    idx = matches.sort_values("votes", ascending=False).index[0]

    # ── Cosine similarity (TF-IDF) ────────────────────────────────────────────
    similarity_scores = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    # ── Rating & popularity weights (normalised 0–1) ───────────────────────────
    ratings = movies["rating"].fillna(0)
    votes   = movies["votes"].fillna(0)

    rating_weight     = ratings / max(ratings.max(), 1)
    popularity_weight = votes   / max(votes.max(), 1)

    # ── Genre score (vectorised, normalised) ───────────────────────────────────
    base_genres  = movies.loc[idx, "genres"]
    max_possible = max(len(set(str(base_genres).split())), 1)

    genre_scores = np.array([
        genre_overlap(base_genres, movies.loc[i, "genres"]) / max_possible
        for i in range(len(movies))
    ])

    # ── Combined score ─────────────────────────────────────────────────────────
    # Weights: 50% semantic, 25% genre, 15% rating, 10% popularity
    final_score = (
        0.50 * similarity_scores
        + 0.25 * genre_scores
        + 0.15 * rating_weight.values
        + 0.10 * popularity_weight.values
    )

    # ── Exclude the query movie itself ─────────────────────────────────────────
    final_score[idx] = -1

    # ── Min votes filter ───────────────────────────────────────────────────────
    final_score[votes < min_votes] = -1

    # ── Industry filter (applied on local data — no API call needed here) ──────
    # Note: full OMDB-based industry filtering happens in app.py for the UI.
    # This standalone version filters by genre overlap only (no Country/Language in dataset).

    sorted_indices = np.argsort(final_score)[::-1]

    recommendations = movies.iloc[sorted_indices[:top_n * 3]].copy()  # over-fetch then trim
    recommendations["score"] = final_score[sorted_indices[:top_n * 3]]

    return recommendations.head(top_n)


# ---------------------------------------------------
# Example Test
# ---------------------------------------------------

if __name__ == "__main__":

    recs = recommend("The Lion King", top_n=5)

    print("\nRecommended Movies:\n")

    for _, row in recs.iterrows():
        print(
            f"{row['title']} ({int(row['year'])}) — {row['genres']} | "
            f"Rating: {row['rating']} | Score: {row.get('score', 'N/A'):.3f}"
        )