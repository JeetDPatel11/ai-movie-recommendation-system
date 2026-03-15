import pandas as pd
import os

print("Loading datasets...")

# Load title basics
basics = pd.read_csv(
    "data/raw/title.basics.tsv",
    sep="\t",
    low_memory=False,
    na_values=["\\N"]  # ✅ FIX: Properly handle \N as NaN instead of string checks later
)

ratings = pd.read_csv(
    "data/raw/title.ratings.tsv",
    sep="\t",
    na_values=["\\N"]
)

print("Filtering only movies...")

# Keep only movies
basics = basics[basics["titleType"] == "movie"]

# ✅ FIX: Use dropna instead of string comparison (\\N already handled above)
basics = basics.dropna(subset=["startYear"])
basics["startYear"] = basics["startYear"].astype(int)

# ✅ NEW: Filter out very old/obscure movies (improves recommendation quality)
basics = basics[basics["startYear"] >= 1950]

print("Merging ratings with movie metadata...")

movies = basics.merge(ratings, on="tconst", how="left")

# Keep important columns
movies = movies[[
    "tconst",
    "primaryTitle",
    "startYear",
    "genres",
    "averageRating",
    "numVotes"
]]

# Rename columns
movies.columns = [
    "movie_id",
    "title",
    "year",
    "genres",
    "rating",
    "votes"
]

# ✅ FIX: Remove movies with missing genres (was checking string "\\N" before — now proper NaN)
movies = movies.dropna(subset=["genres"])

# ✅ NEW: Remove movies with zero votes (no data to rank by)
movies = movies[movies["votes"].notna() & (movies["votes"] > 0)]

# ✅ NEW: Remove duplicate titles (keep highest voted version)
movies = movies.sort_values("votes", ascending=False).drop_duplicates(subset=["title"], keep="first")

movies = movies.reset_index(drop=True)

print("Total movies after cleaning:", len(movies))

# ✅ FIX: Ensure output directory exists before saving
os.makedirs("data/processed", exist_ok=True)

movies.to_csv(
    "data/processed/movies_cleaned.csv",
    index=False
)

print("Dataset saved to data/processed/movies_cleaned.csv")