import streamlit as st
import pandas as pd
import pickle
import requests
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="AI Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
        border: 1px solid #3a3a5c;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 8px 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #ff6b6b;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #a0a0c0;
        margin-top: 4px;
    }
    .score-label {
        font-size: 0.8rem;
        color: #a0a0c0;
        margin-bottom: 2px;
    }
    .formula-box {
        background: #1e1e2e;
        border-left: 4px solid #ff6b6b;
        border-radius: 8px;
        padding: 16px 20px;
        font-family: monospace;
        font-size: 0.95rem;
        margin: 12px 0;
    }
    .tag {
        display: inline-block;
        background: #2a2a4e;
        border: 1px solid #4a4a7e;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.8rem;
        color: #a0a0ff;
        margin: 3px;
    }
    .why-box {
        background: #1a2a1a;
        border-left: 3px solid #4caf50;
        border-radius: 6px;
        padding: 10px 14px;
        font-size: 0.85rem;
        color: #90c090;
        margin-top: 8px;
    }
    .about-section {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 24px;
        margin: 12px 0;
        border: 1px solid #3a3a5c;
    }
    .pipeline-step {
        background: #2a2a3e;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 8px 0;
        border-left: 3px solid #ff6b6b;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

@st.cache_data
def load_data():
    movies = pd.read_pickle("models/movies_metadata.pkl")

    with open("models/semantic_embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)

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
    try:
        url = f"http://www.omdbapi.com/?t={requests.utils.quote(title)}&apikey={API_KEY}"
        response = requests.get(url, timeout=3)
        data = response.json()
        if data.get("Response") == "True":
            return data
    except Exception:
        pass
    return None


# -------------------------------------------------
# INDUSTRY FILTER
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
# -------------------------------------------------

def genre_overlap(g1, g2) -> int:
    if pd.isna(g1) or pd.isna(g2):
        return 0
    return len(set(str(g1).split()).intersection(set(str(g2).split())))


# -------------------------------------------------
# RECOMMENDER
# -------------------------------------------------

def recommend(movie_display: str, industry: str = "All", top_n: int = 5):

    movie_row = movies[movies["display_title"] == movie_display]
    if movie_row.empty:
        return [], None

    idx = movie_row.index[0]

    similarity_scores = cosine_similarity(
        [embeddings[idx]], embeddings
    ).flatten()

    ratings = movies["rating"].fillna(0)
    votes   = movies["votes"].fillna(0)

    rating_weight     = ratings / max(ratings.max(), 1)
    popularity_weight = votes   / max(votes.max(), 1)

    base_genres  = movies.loc[idx, "genres"]
    max_possible = max(len(set(str(base_genres).split())), 1)

    genre_scores = np.array([
        genre_overlap(base_genres, movies.loc[i, "genres"]) / max_possible
        for i in range(len(movies))
    ])

    scores = (
        0.50 * similarity_scores
        + 0.25 * genre_scores
        + 0.15 * rating_weight.values
        + 0.10 * popularity_weight.values
    )

    scores[idx] = -1
    sorted_indices = np.argsort(scores)[::-1]

    recommendations = []
    progress = st.progress(0, text="Fetching movie details…")
    checked  = 0

    for i in sorted_indices[:50]:
        if len(recommendations) >= top_n:
            break

        row        = movies.iloc[i]
        movie_data = fetch_movie_details(row["title"])

        checked += 1
        progress.progress(min(checked / 50, 1.0), text="Fetching movie details…")

        if movie_data is None:
            continue
        if not filter_by_industry(movie_data, industry):
            continue

        shared_genres = set(str(base_genres).split()).intersection(
            set(str(row["genres"]).split())
        )
        shared_genres_str = ", ".join(shared_genres) if shared_genres else "None"

        recommendations.append({
            "row":           row,
            "omdb":          movie_data,
            "score":         round(scores[i] * 100, 1),
            "sem_score":     round(similarity_scores[i] * 100, 1),
            "genre_score":   round(genre_scores[i] * 100, 1),
            "shared_genres": shared_genres_str,
        })

    progress.empty()
    return recommendations, movies.loc[idx]


# -------------------------------------------------
# NAVIGATION
# -------------------------------------------------

st.sidebar.title("🎬 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🎯 Recommend", "📊 Evaluation", "ℹ️ About"]
)


# =================================================
# PAGE 1 — RECOMMEND
# =================================================

if page == "🎯 Recommend":

    st.title("🎬 AI Movie Recommendation System")
    st.caption("Powered by semantic embeddings + IMDb data")

    with st.sidebar:
        st.markdown("---")
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
        min_rating = st.slider("Minimum IMDb Rating", min_value=0.0, max_value=9.0, value=5.0, step=0.5)

    movie_list     = movies["display_title"].dropna().unique()
    selected_movie = st.selectbox(
        "🎥 Select a movie to get recommendations",
        sorted(movie_list)
    )

    if st.button("🍿 Recommend", use_container_width=True):

        with st.spinner("Finding similar movies…"):
            results, base_movie = recommend(selected_movie, industry=industry, top_n=top_n)

        if not results:
            st.warning("No recommendations found. Try changing the filters.")
        else:
            st.subheader(f"🎯 Top {len(results)} Recommendations for *{selected_movie}*")

            for item in results:

                row        = item["row"]
                movie_data = item["omdb"]
                score      = item["score"]
                sem_score  = item["sem_score"]
                genre_sc   = item["genre_score"]
                shared     = item["shared_genres"]

                movie_country = movie_data.get("Country", "")
                if country_filter != "All" and country_filter not in movie_country:
                    continue

                try:
                    imdb_rating = float(movie_data.get("imdbRating", 0))
                except (ValueError, TypeError):
                    imdb_rating = 0.0

                if imdb_rating < min_rating:
                    continue

                col1, col2 = st.columns([1, 3])

                with col1:
                    poster = movie_data.get("Poster", "N/A")
                    if poster and poster != "N/A":
                        st.image(poster, width=160)
                    else:
                        st.markdown("🎞️ *No poster*")

                with col2:
                    st.markdown(f"### {movie_data['Title']} ({movie_data.get('Year','?')})")

                    m1, m2, m3 = st.columns(3)
                    m1.metric("⭐ IMDb", movie_data.get("imdbRating", "N/A"))
                    m2.metric("⏱️ Runtime", movie_data.get("Runtime", "N/A"))
                    m3.metric("📅 Year", movie_data.get("Year", "N/A"))

                    st.write(f"🌍 **Country:** {movie_data.get('Country','N/A')}")
                    st.write(f"🎭 **Genre:** {movie_data.get('Genre','N/A')}")
                    st.write(f"🎬 **Director:** {movie_data.get('Director','N/A')}")
                    st.write(f"🌟 **Cast:** {movie_data.get('Actors','N/A')}")
                    st.write(f"📝 **Plot:** {movie_data.get('Plot','N/A')}")

                    # Match score bar
                    st.markdown(f"<p class='score-label'>🎯 Overall Match Score: {score}%</p>", unsafe_allow_html=True)
                    st.progress(min(score / 100, 1.0))

                    # Why recommended
                    st.markdown(
                        f"""<div class='why-box'>
                        ✅ <b>Why recommended:</b> Semantic similarity <b>{sem_score}%</b> ·
                        Genre match <b>{genre_sc}%</b> · Shared genres: <b>{shared}</b>
                        </div>""",
                        unsafe_allow_html=True
                    )

                st.markdown("---")


# =================================================
# PAGE 2 — EVALUATION
# =================================================

elif page == "📊 Evaluation":

    st.title("📊 Model Evaluation & Statistics")
    st.caption("How the recommendation engine works and performs")

    # Dataset stats
    st.markdown("### 🗄️ Dataset Statistics")

    total_movies = len(movies)
    year_min     = int(movies["year"].replace(0, np.nan).dropna().min())
    year_max     = int(movies["year"].replace(0, np.nan).dropna().max())
    avg_rating   = round(movies["rating"].dropna().mean(), 2)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"""<div class='metric-card'>
        <div class='metric-value'>{total_movies:,}</div>
        <div class='metric-label'>Total Movies</div>
    </div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class='metric-card'>
        <div class='metric-value'>{year_min}–{year_max}</div>
        <div class='metric-label'>Year Range</div>
    </div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class='metric-card'>
        <div class='metric-value'>{avg_rating}</div>
        <div class='metric-label'>Avg IMDb Rating</div>
    </div>""", unsafe_allow_html=True)
    c4.markdown(f"""<div class='metric-card'>
        <div class='metric-value'>384</div>
        <div class='metric-label'>Embedding Dimensions</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Scoring formula
    st.markdown("### ⚙️ Scoring Formula")
    st.markdown("Each candidate movie is scored using a **weighted hybrid formula** combining four signals:")
    st.markdown("""<div class='formula-box'>
    Final Score = 0.50 × Semantic Similarity<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    + 0.25 × Genre Overlap<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    + 0.15 × IMDb Rating Weight<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    + 0.10 × Popularity (Vote Count)
    </div>""", unsafe_allow_html=True)

    st.markdown("#### Signal Weights Breakdown")
    weight_df = pd.DataFrame({
        "Signal":  ["Semantic Similarity", "Genre Overlap", "IMDb Rating", "Popularity"],
        "Weight %": [50, 25, 15, 10]
    })
    st.bar_chart(weight_df.set_index("Signal"))

    st.markdown("---")

    # Model info
    st.markdown("### 🧠 AI Model Information")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Semantic Embedding Model")
        st.markdown("""
        | Property | Value |
        |---|---|
        | Model | all-MiniLM-L6-v2 |
        | Developer | Microsoft |
        | Downloaded from | HuggingFace |
        | Model size | ~90 MB |
        | Embeddings file size | ~399 MB (272k movies) |
        | Embedding size | 384 dimensions per movie |
        | Input | Title + year + genres |
        """)

    with col2:
        st.markdown("#### TF-IDF Vectorizer")
        st.markdown("""
        | Property | Value |
        |---|---|
        | Method | TF-IDF |
        | Features | Genre text |
        | Vocabulary | 287 terms |
        | N-gram range | (1, 2) |
        | Similarity metric | Cosine similarity |
        | Purpose | Genre-based matching |
        """)

    st.markdown("---")

    # Live similarity test
    st.markdown("### 🎯 Live Similarity Test")
    st.caption("Select any movie to see its top 5 most similar movies by semantic score only")

    test_movie = st.selectbox(
        "Select a movie to test",
        sorted(movies["display_title"].dropna().unique()),
        key="eval_test"
    )

    if st.button("Run Similarity Test"):
        movie_row = movies[movies["display_title"] == test_movie]
        if not movie_row.empty:
            idx        = movie_row.index[0]
            sim_scores = cosine_similarity([embeddings[idx]], embeddings).flatten()
            sim_scores[idx] = -1
            top5_idx   = np.argsort(sim_scores)[::-1][:5]

            st.markdown("#### Top 5 Most Similar Movies (Semantic Score Only)")
            for rank, i in enumerate(top5_idx, 1):
                score = round(float(sim_scores[i]) * 100, 1)
                title = movies.iloc[i]["title"]
                genre = movies.iloc[i]["genres"]
                year  = int(movies.iloc[i]["year"])
                st.markdown(f"**{rank}. {title} ({year})**")
                st.markdown(f"<p class='score-label'>Genres: {genre} · Semantic score: {score}%</p>", unsafe_allow_html=True)
                st.progress(float(min(score / 100, 1.0)))

    st.markdown("---")

    # Genre distribution
    st.markdown("### 🎭 Top 10 Genres in Dataset")
    genre_series = movies["genres"].dropna().str.split().explode()
    top_genres   = genre_series.value_counts().head(10).reset_index()
    top_genres.columns = ["Genre", "Count"]
    st.bar_chart(top_genres.set_index("Genre"))

    st.markdown("---")

    # Rating distribution
    st.markdown("### ⭐ IMDb Rating Distribution")
    rating_bins = pd.cut(
        movies["rating"].dropna(),
        bins=[0,2,4,6,7,8,9,10],
        labels=["0-2","2-4","4-6","6-7","7-8","8-9","9-10"]
    )
    st.bar_chart(rating_bins.value_counts().sort_index())


# =================================================
# PAGE 3 — ABOUT
# =================================================

elif page == "ℹ️ About":

    st.title("ℹ️ About This Project")

    st.markdown("""<div class='about-section'>
        <h3>🎬 AI Movie Recommendation System</h3>
        <p>A content-based movie recommendation system that uses <b>semantic embeddings</b>
        and <b>TF-IDF vectorization</b> to find movies similar to any given movie from a
        database of <b>272,393 IMDb movies</b>.</p>
        <p>Built as a <b>Final Year Project</b> for Bachelor of Engineering in
        Computer Engineering — 8th Semester.</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("### 🛠️ Technology Stack")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Frontend**")
        st.markdown("<span class='tag'>Streamlit</span> <span class='tag'>Python</span>", unsafe_allow_html=True)
        st.markdown("**Data**")
        st.markdown("<span class='tag'>IMDb Datasets</span> <span class='tag'>OMDB API</span> <span class='tag'>Pandas</span>", unsafe_allow_html=True)

    with col2:
        st.markdown("**AI / ML**")
        st.markdown("<span class='tag'>Sentence Transformers</span> <span class='tag'>scikit-learn</span> <span class='tag'>NumPy</span>", unsafe_allow_html=True)
        st.markdown("**Model**")
        st.markdown("<span class='tag'>all-MiniLM-L6-v2</span> <span class='tag'>TF-IDF</span> <span class='tag'>Cosine Similarity</span>", unsafe_allow_html=True)

    with col3:
        st.markdown("**Deployment**")
        st.markdown("<span class='tag'>Streamlit Cloud</span> <span class='tag'>GitHub</span> <span class='tag'>Git LFS</span>", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### ⚙️ System Pipeline")
    steps = [
        ("1️⃣ Data Collection",     "Downloaded IMDb title.basics.tsv and title.ratings.tsv datasets containing 272,393 movies from 1950 onwards."),
        ("2️⃣ Data Preprocessing",  "Filtered movies only, removed missing genres and zero-vote entries, deduplicated by highest vote count."),
        ("3️⃣ Feature Engineering", "Applied TF-IDF vectorization on genre text with bigram support (287 vocabulary terms)."),
        ("4️⃣ Semantic Embeddings", "Used all-MiniLM-L6-v2 Sentence Transformer to encode each movie into a 384-dimensional vector capturing semantic meaning."),
        ("5️⃣ Recommendation",      "Combined semantic similarity (50%), genre overlap (25%), IMDb rating (15%), and popularity (10%) into a final score."),
        ("6️⃣ Deployment",          "Deployed on Streamlit Cloud with model files stored via Git LFS on GitHub."),
    ]
    for title, desc in steps:
        st.markdown(f"""<div class='pipeline-step'>
            <b>{title}</b><br>
            <span style='color:#a0a0c0;font-size:0.9rem'>{desc}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### ⚠️ Limitations & Future Work")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Current Limitations**")
        st.markdown("""
        - OMDB API limited to 1,000 requests/day
        - No user login or personalization
        - Cold start problem on Streamlit free tier
        - Content-based only (no collaborative filtering)
        """)

    with col2:
        st.markdown("**Future Improvements**")
        st.markdown("""
        - Add collaborative filtering using user ratings
        - Include movie trailers via YouTube API
        - Add user accounts and watch history
        - Implement A/B testing for recommendation quality
        """)

    st.markdown("---")

    st.markdown("### 👨‍💻 Developer")
    st.markdown("""<div class='about-section'>
        <b>Jeet Patel</b><br>
        <span style='color:#a0a0c0'>B.E. Computer Engineering — 8th Semester</span>
    </div>""", unsafe_allow_html=True)