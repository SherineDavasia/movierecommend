import pandas as pd 
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
ratings_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv(r'ml-100k/u.data', sep='\t', names=ratings_cols)

movies_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url',
               'unknown', 'Action', 'Adventure', 'Animation', 'Children',
                'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
                'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

movies = pd.read_csv(r'ml-100k/u.item', sep='|', names=movies_cols, encoding='latin-1')

# Merge
combined = pd.merge(ratings, movies[['movie_id', 'title']], on='movie_id')
movie_user_matrix = combined.pivot_table(index='title', columns='user_id', values='rating').fillna(0)
similarity_matrix = cosine_similarity(movie_user_matrix)
similarity_df = pd.DataFrame(similarity_matrix, index=movie_user_matrix.index, columns=movie_user_matrix.index)

# Streamlit App Design
st.set_page_config(page_title="🎬 Movie Recommender", layout="centered")

st.title("🎬 Movie Recommender Engine")
st.markdown("""
<style>
body {
    background-color: #fdf6f6;
}
.stApp {
    font-family: 'Segoe UI', sans-serif;
}
</style>
""", unsafe_allow_html=True)

st.subheader("✨ Find movies similar to your favorite picks!")
movie_list = list(similarity_df.columns)
movie_input = st.selectbox("🎞️ Pick a movie you liked:", sorted(movie_list))

if st.button("🎯 Recommend Similar Movies"):
    st.markdown(f"Here are top picks because you liked **{movie_input}**:")
    similar_movies = similarity_df[movie_input].sort_values(ascending=False)[1:6]
    for i, (title, score) in enumerate(similar_movies.items(), start=1):
        st.write(f"{i}. 🎥 **{title}** — _(Similarity Score: {score:.2f})_")
