import streamlit as st
import pandas as pd
import pickle
import requests
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import gdown

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 AI Movie Recommendation System")
st.caption("Powered by semantic embeddings + IMDb data")

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

@st.cache_data
def load_data():

    os.makedirs("models", exist_ok=True)

    movies_path = "models/movies_metadata.pkl"
    embeddings_path = "models/semantic_embeddings.pkl"

    # ---- Check movies file ----
    if not os.path.exists(movies_path):
        st.error("movies_metadata.pkl not found in models folder.")
        st.stop()

    # ---- Download embeddings if missing ----
    if not os.path.exists(embeddings_path):

        st.info("Downloading embeddings file. Please wait...")

        file_id = "13k3As0EXIw4eBw6QrdiyXuVwkbztJOv-"
        url = f"https://drive.google.com/uc?id={file_id}"

        gdown.download(url, embeddings_path, quiet=False)

    # ---- Load data ----
    movies = pd.read_pickle(movies_path)

    with open(embeddings_path, "rb") as f:
        embeddings = pickle.load(f)

    # ---- Data cleaning ----
    movies["year"] = movies["year"].fillna(0).astype(int)
    movies["display_title"] = movies["title"] + " (" + movies["year"].astype(str) + ")"

    return movies, embeddings


movies, embeddings = load_data()

# -------------------------------------------------
# OMDB API
# -------------------------------------------------

API_KEY = "99b21769"


@st.cache_data(ttl=86400)
def fetch_movie_details(title: str):
    """Fetch movie metadata from OMDB. Returns dict or None."""
    try:
        url = f"http://www.omdbapi.com/?t={requests.utils.quote(title)}&apikey={API_KEY}"
        response = requests.get(url, timeout=5)
        data = response.json()
        if data.get("Response") == "True":
            return data
    except Exception:
        pass
    return None


# -------------------------------------------------
# INDUSTRY FILTER  (on OMDB data dict)
# ✅ FIX: Use "in" checks — OMDB returns comma-separated strings
# -------------------------------------------------

def filter_by_industry(movie_data: dict, industry: str) -> bool:

    if industry == "All":
        return True

    country  = movie_data.get("Country", "")
    language = movie_data.get("Language", "")

    industry_map = {
        "Bollywood":  ("India", "Hindi"),
        "Tollywood":  ("India", "Telugu"),
        "Kollywood":  ("India", "Tamil"),
        "Mollywood":  ("India", "Malayalam"),
        "Sandalwood": ("India", "Kannada"),
    }

    if industry in industry_map:
        req_country, req_lang = industry_map[industry]
        return req_country in country and req_lang in language

    if industry == "Hollywood":
        return any(c in country for c in ["USA", "United States", "UK", "Canada", "Australia"])

    return True


# -------------------------------------------------
# GENRE OVERLAP
# ✅ FIX: Split on space (genres stored space-separated after feature_engineering)
# -------------------------------------------------

def genre_overlap(g1, g2) -> int:
    if pd.isna(g1) or pd.isna(g2):
        return 0
    return len(set(str(g1).split()).intersection(set(str(g2).split())))


# -------------------------------------------------
# RECOMMENDER
# ✅ IMPROVEMENTS:
#   - Normalised genre score
#   - Vectorised score computation
#   - Progress bar while fetching OMDB results
#   - Graceful empty-result message
# -------------------------------------------------

def recommend(movie_display: str, industry: str = "All", top_n: int = 5) -> pd.DataFrame:

    movie_row = movies[movies["display_title"] == movie_display]

    if movie_row.empty:
        return pd.DataFrame()

    idx = movie_row.index[0]

    # Semantic similarity
    similarity_scores = cosine_similarity(
        [embeddings[idx]], embeddings
    ).flatten()

    # Rating & popularity weights
    ratings = movies["rating"].fillna(0)
    votes   = movies["votes"].fillna(0)

    rating_weight     = ratings / max(ratings.max(), 1)
    popularity_weight = votes   / max(votes.max(), 1)

    # Genre score (normalised)
    base_genres  = movies.loc[idx, "genres"]
    max_possible = max(len(set(str(base_genres).split())), 1)

    genre_scores = np.array([
        genre_overlap(base_genres, movies.loc[i, "genres"]) / max_possible
        for i in range(len(movies))
    ])

    # Combined weighted score
    scores = (
        0.50 * similarity_scores
        + 0.25 * genre_scores
        + 0.15 * rating_weight.values
        + 0.10 * popularity_weight.values
    )

    scores[idx] = -1  # exclude query movie

    sorted_indices = np.argsort(scores)[::-1]

    recommendations = []

    # ✅ NEW: Show a progress bar while we query OMDB
    progress = st.progress(0, text="Fetching movie details…")
    checked  = 0

    for i in sorted_indices:

        if len(recommendations) >= top_n:
            break

        row        = movies.iloc[i]
        movie_data = fetch_movie_details(row["title"])

        checked += 1
        progress.progress(min(checked / (top_n * 6), 1.0), text="Fetching movie details…")

        if movie_data is None:
            continue

        if not filter_by_industry(movie_data, industry):
            continue

        recommendations.append({"row": row, "omdb": movie_data})

    progress.empty()

    return recommendations  # list of dicts with "row" and "omdb" keys


# -------------------------------------------------
# SIDEBAR FILTERS  (✅ moved to sidebar — cleaner layout)
# -------------------------------------------------

with st.sidebar:
    st.header("🔧 Filters")

    industry = st.selectbox(
        "Film Industry",
        ["All", "Hollywood", "Bollywood", "Tollywood", "Kollywood", "Mollywood", "Sandalwood"]
    )

    country_filter = st.selectbox(
        "Country",
        ["All", "India", "USA", "UK", "South Korea", "Japan"]
    )

    top_n = st.slider("Number of Recommendations", min_value=3, max_value=15, value=5)

    # ✅ NEW: Min IMDb rating filter
    min_rating = st.slider("Minimum IMDb Rating", min_value=0.0, max_value=9.0, value=5.0, step=0.5)

# -------------------------------------------------
# MOVIE SELECTOR
# -------------------------------------------------

movie_list = movies["display_title"].dropna().unique()

selected_movie = st.selectbox(
    "🎥 Select a movie to get recommendations",
    sorted(movie_list)
)

# -------------------------------------------------
# RECOMMEND BUTTON
# -------------------------------------------------

if st.button("🍿 Recommend", use_container_width=True):

    with st.spinner("Finding similar movies…"):
        results = recommend(selected_movie, industry=industry, top_n=top_n)

    if not results:
        st.warning("No recommendations found. Try changing the filters.")
    else:
        st.subheader(f"🎯 Top {len(results)} Recommendations for *{selected_movie}*")

        for item in results:

            row        = item["row"]
            movie_data = item["omdb"]

            # Country filter
            movie_country = movie_data.get("Country", "")
            if country_filter != "All" and country_filter not in movie_country:
                continue

            # ✅ NEW: Rating filter
            try:
                imdb_rating = float(movie_data.get("imdbRating", 0))
            except (ValueError, TypeError):
                imdb_rating = 0.0

            if imdb_rating < min_rating:
                continue

            # ── Card layout ───────────────────────────────────────────────────
            col1, col2 = st.columns([1, 3])

            with col1:
                poster = movie_data.get("Poster", "N/A")
                if poster and poster != "N/A":
                    st.image(poster, width=160)
                else:
                    st.markdown("🎞️ *No poster*")

            with col2:
                st.markdown(f"### {movie_data['Title']} ({movie_data.get('Year','?')})")

                # ✅ NEW: Inline metrics row
                m1, m2, m3 = st.columns(3)
                m1.metric("⭐ IMDb", movie_data.get("imdbRating", "N/A"))
                m2.metric("⏱️ Runtime", movie_data.get("Runtime", "N/A"))
                m3.metric("📅 Year", movie_data.get("Year", "N/A"))

                st.write(f"🌍 **Country:** {movie_data.get('Country','N/A')}")
                st.write(f"🎭 **Genre:** {movie_data.get('Genre','N/A')}")
                st.write(f"🎬 **Director:** {movie_data.get('Director','N/A')}")
                st.write(f"🌟 **Cast:** {movie_data.get('Actors','N/A')}")
                st.write(f"📝 **Plot:** {movie_data.get('Plot','N/A')}")

            st.markdown("---")