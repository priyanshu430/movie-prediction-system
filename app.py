from flask import Flask, request
import pickle
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# -------------------- LOAD PICKLES --------------------
movies = pickle.load(open('movies.pkl', 'rb'))  # DataFrame with movie info
vectors = pickle.load(open('vectors.pkl', 'rb'))  # TF-IDF vectors (dense)
similarity = pickle.load(open('similarity.pkl', 'rb'))  # Cosine similarity matrix


# -------------------- HELPER FUNCTION TO FETCH POSTER --------------------
def fetch_poster(movie_id):
    api_key = "YOUR_TMDB_API_KEY"  # Replace with your TMDB API key
    url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key=cfcaf405e5a601152ad8f9ef81532856'
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        if data.get('poster_path'):
            return "https://image.tmdb.org/t/p/w500/" + data['poster_path']
    except:
        pass
    # fallback image
    return "https://via.placeholder.com/200x300?text=No+Image"


# -------------------- MOVIE RECOMMENDER FUNCTION --------------------
def recommend_movies(movie_name, top_n=5):
    movie_name = movie_name.lower().strip()
    matches = movies[movies['title'].str.lower().str.contains(movie_name)]

    if matches.empty:
        return [], []

    idx = matches.index[0]
    movie_vector = vectors[idx].reshape(1, -1)

    sim_scores = cosine_similarity(movie_vector, vectors)
    top_indices = np.argsort(sim_scores[0])[::-1][1:top_n + 1]

    recommended_titles = []
    recommended_posters = []
    for i in top_indices:
        recommended_titles.append(movies.iloc[i]['title'])
        recommended_posters.append(fetch_poster(movies.iloc[i]['id']))

    return recommended_titles, recommended_posters


# -------------------- ROUTE WITH EMBEDDED HTML --------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    html = """
    <html>
    <head>
        <title>Movie Recommender</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; padding: 20px; }
            .movie { display: inline-block; margin: 10px; }
            .movie img { width: 200px; height: 300px; }
            input { padding: 5px; width: 300px; }
            button { padding: 5px 10px; }
        </style>
    </head>
    <body>
        <h1>Movie Recommender System</h1>
        <form method="POST">
            <input type="text" name="movie_name" placeholder="Enter movie name" required>
            <button type="submit">Search</button>
        </form>
    """
    if request.method == 'POST':
        movie_name = request.form.get('movie_name')
        titles, posters = recommend_movies(movie_name)

        if titles:
            html += f"<h2>Recommendations for '{movie_name}'</h2>"
            for title, poster in zip(titles, posters):
                html += f"""
                    <div class='movie'>
                        <p>{title}</p>
                        <img src='{poster}' alt='{title}'>
                    </div>
                """
        else:
            html += f"<p>No movies found with name '{movie_name}'</p>"

    html += "</body></html>"
    return html


# -------------------- RUN APP --------------------
if __name__ == '__main__':
    app.run(debug=True)
