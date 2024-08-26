import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import tkinter as tk
from tkinter import simpledialog, messagebox

# Carregar os dados de avaliações e filmes
ratings = pd.read_csv('ratings.csv') 
movies = pd.read_csv('movies.csv')    

# Exibir as primeiras linhas dos datasets
print(ratings.head())
print(movies.head())

# Verificar se há valores faltantes
print(ratings.isnull().sum())
print(movies.isnull().sum())

# Verificar se há valores duplicados
print(ratings.duplicated().sum())
print(movies.duplicated().sum())

# Remover valores duplicados
ratings = ratings.drop_duplicates()
movies = movies.drop_duplicates()

# Normalizar as avaliações
ratings['rating'] = ratings['rating'] / ratings['rating'].max()

# Dividir os dados: 80% para treino, 20% para teste
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

# Criar um dicionário de IDs de filmes para títulos
movie_dict = movies.set_index('movieId')['title'].to_dict()

# Criar a matriz usuário-filme
user_movie_matrix = train_data.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Treinar o modelo KNN
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(user_movie_matrix)

# Escolher um usuário para fazer a previsão
user_id = 1  # Exemplo de usuário
user_vector = user_movie_matrix.loc[user_id].values.reshape(1, -1)

# Encontrar os vizinhos mais próximos
distances, indices = model_knn.kneighbors(user_vector, n_neighbors=10)

# Calcular as recomendações (média ponderada das avaliações dos vizinhos)
neighbors_ratings = user_movie_matrix.iloc[indices.flatten()].values
predicted_ratings = neighbors_ratings.mean(axis=0)

# Recomendar filmes com base nas avaliações previstas mais altas
recommended_movie_ids = user_movie_matrix.columns[predicted_ratings.argsort()[::-1]]
recommended_movie_titles = [movie_dict[movie_id] for movie_id in recommended_movie_ids[:10]]

print("Recomendações para o usuário", user_id, ":")
print(recommended_movie_titles)

# Criar a matriz usuário-filme para o conjunto de teste
test_user_movie_matrix = test_data.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Função para prever avaliações para o conjunto de teste
def predict_ratings(user_movie_matrix, model_knn):
    predictions = []
    for user_id in user_movie_matrix.index:
        user_vector = user_movie_matrix.loc[user_id].values.reshape(1, -1)
        distances, indices = model_knn.kneighbors(user_vector, n_neighbors=10)
        neighbors_ratings = user_movie_matrix.iloc[indices.flatten()].values
        predicted_ratings = neighbors_ratings.mean(axis=0)
        predictions.append(predicted_ratings)
    return np.array(predictions)

# Prever avaliações para o conjunto de teste
test_user_movie_matrix = test_data.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Ajustar para apenas prever para usuários que estão no conjunto de teste
predictions = predict_ratings(test_user_movie_matrix, model_knn)

# Garantir que y_true e y_pred têm o mesmo comprimento
y_true = test_user_movie_matrix.values.flatten()
y_pred = np.zeros_like(y_true)  # Inicializar y_pred com zeros

# Mapear as previsões para as posições corretas
for i, user_id in enumerate(test_user_movie_matrix.index):
    if user_id in user_movie_matrix.index:
        user_vector = user_movie_matrix.loc[user_id].values.reshape(1, -1)
        distances, indices = model_knn.kneighbors(user_vector, n_neighbors=10)
        neighbors_ratings = user_movie_matrix.iloc[indices.flatten()].values
        predicted_ratings = neighbors_ratings.mean(axis=0)
        movie_ids = user_movie_matrix.columns
        for movie_id in movie_ids:
            if movie_id in test_user_movie_matrix.columns:
                movie_index = test_user_movie_matrix.columns.get_loc(movie_id)
                y_pred[i * len(movie_ids) + movie_index] = predicted_ratings[movie_ids.get_loc(movie_id)]

# Verificar tamanho antes de calcular o RMSE
print("y_true length:", len(y_true))
print("y_pred length:", len(y_pred))

# Calcular o RMSE
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print("RMSE:", rmse)

# Ajustar o número de vizinhos
model_knn = NearestNeighbors(n_neighbors=15, metric='cosine', algorithm='brute')
model_knn.fit(user_movie_matrix)

# Criar recomendação baseada em conteúdo
# Criar uma matriz TF-IDF para os gêneros dos filmes
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Calcular similaridade de cosseno entre os filmes
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Função para recomendar filmes baseados em conteúdo
def recommend_movies_based_on_content(movie_title, cosine_sim=cosine_sim):
    idx = movies.index[movies['title'] == movie_title].tolist()
    if not idx:
        return pd.Series()
    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Excluir o próprio filme
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

# Exemplo de recomendação baseada em conteúdo
print(recommend_movies_based_on_content('Toy Story (1995)'))

# Criar a interface de usuário com Tkinter
# Função para mostrar recomendações
def show_recommendations():
    movie_title = simpledialog.askstring("Input", "Enter movie title:")
    if not movie_title:
        messagebox.showwarning("Input Error", "You must enter a movie title.")
        return

    recommendations = recommend_movies_based_on_content(movie_title)
    if recommendations.empty:
        result_text = "No recommendations found."
    else:
        result_text = "Recommended Movies:\n" + "\n".join(recommendations)
        
    result_label.config(text=result_text)

# Configuração da janela principal
root = tk.Tk()
root.title("Movie Recommendation System")

# Botão para obter recomendações
recommend_button = tk.Button(root, text="Get Recommendations", command=show_recommendations)
recommend_button.pack(pady=20)

# Rótulo para mostrar as recomendações
result_label = tk.Label(root, text="", justify="left")
result_label.pack(padx=20, pady=20)

# Iniciar o loop da interface
root.mainloop()
