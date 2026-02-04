import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Netflix Analytics Dashboard", layout="wide")

st.title("🎬 Netflix Movies Analytics & IMDb Rating Predictor")

# ================= LOAD DATA =================
data = pd.read_csv("clean_netflix_data.csv")

# ================= LOAD MODEL =================
model = joblib.load("imdb_rating_predictor.pkl")

# ================= SIDEBAR =================
st.sidebar.title("🔍 Filters & Info")

year_filter = st.sidebar.slider(
    "Select Release Year",
    int(data['release_year'].min()),
    int(data['release_year'].max()),
    (2010, 2020)
)

type_filter = st.sidebar.multiselect(
    "Select Type",
    options=data['type'].unique(),
    default=data['type'].unique()
)

st.sidebar.markdown("## 📌 About This Project")
st.sidebar.info(
    "This dashboard analyzes Netflix titles and predicts IMDb ratings using a Machine Learning model. "
    "Built with Python, Pandas, Scikit-learn and Streamlit."
)

# ================= FILTER DATA =================
filtered_data = data[
    (data['release_year'] >= year_filter[0]) &
    (data['release_year'] <= year_filter[1]) &
    (data['type'].isin(type_filter))
]

# ================= DATA OVERVIEW =================
st.header("📊 Dataset Overview")
st.write("Total Titles:", filtered_data.shape[0])
st.dataframe(filtered_data.head())

# ================= KEY METRICS =================
st.header("📌 Key Metrics")

col1, col2, col3 = st.columns(3)
col1.metric("Average IMDb Score", round(filtered_data['imdb_score'].mean(), 2))
col2.metric("Average Runtime", int(filtered_data['runtime'].mean()))
col3.metric("Average TMDB Score", round(filtered_data['tmdb_score'].mean(), 2))

# ================= VISUALIZATIONS =================
st.header("📈 Visual Analysis")

col4, col5 = st.columns(2)

with col4:
    st.subheader("IMDb Score Distribution")
    fig, ax = plt.subplots()
    sns.histplot(filtered_data['imdb_score'], bins=20, kde=True, ax=ax)
    st.pyplot(fig)

with col5:
    st.subheader("Runtime vs IMDb Score")
    fig, ax = plt.subplots()
    sns.scatterplot(x=filtered_data['runtime'], y=filtered_data['imdb_score'], ax=ax)
    st.pyplot(fig)

# ================= YEAR TREND =================
st.subheader("📅 Average IMDb Score by Release Year")
year_trend = filtered_data.groupby("release_year")["imdb_score"].mean()

fig, ax = plt.subplots()
year_trend.plot(ax=ax)
st.pyplot(fig)

# ================= TOP GENRES =================
st.subheader("🎭 Most Common Genres")
genre_counts = filtered_data['genres'].str.split(', ').explode().value_counts().head(10)

fig, ax = plt.subplots()
genre_counts.plot(kind='bar', ax=ax)
st.pyplot(fig)

# ================= TOP RATED MOVIES =================
st.subheader("🏆 Top Rated Titles")
top_movies = filtered_data.sort_values(by="imdb_score", ascending=False)[
    ["title", "release_year", "imdb_score"]
].head(10)

st.dataframe(top_movies)

# ================= MODEL PERFORMANCE =================
st.header("📊 Model Performance")

col6, col7, col8 = st.columns(3)
col6.metric("R² Score", "0.50")
col7.metric("MAE", "0.54")
col8.metric("RMSE", "0.54")

# ================= PREDICTION SECTION =================
st.header("🎯 Predict IMDb Rating")

runtime = st.slider("Runtime (minutes)", 30, 200, 100)
votes = st.number_input("IMDb Votes", min_value=100, max_value=300000, value=5000)
popularity = st.slider("TMDB Popularity", 0.0, 50.0, 10.0)
tmdb_score = st.slider("TMDB Score", 1.0, 10.0, 6.0)
seasons = st.slider("Number of Seasons (0 for movies)", 0, 10, 1)
type_input = st.selectbox("Type", data['type'].unique())

if st.button("Predict Rating"):
    input_data = pd.DataFrame(
        [[runtime, votes, popularity, tmdb_score, seasons, type_input]],
        columns=['runtime', 'imdb_votes', 'tmdb_popularity', 'tmdb_score', 'seasons', 'type']
    )

    prediction = model.predict(input_data)[0]
    st.success(f"⭐ Predicted IMDb Rating: {round(prediction, 2)}")
