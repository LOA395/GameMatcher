import streamlit as st
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from sklearn.preprocessing import normalize
from pinecone import Pinecone, ServerlessSpec
from googletrans import Translator 

# Cargar el modelo y los datos
@st.cache_resource
def cargar_modelo_y_datos():
    # Cargar el modelo Doc2Vec (asegúrate de guardar y cargar el modelo previamente entrenado)
    model = Doc2Vec.load("app\model_doc2vec.bin")
    
    # Cargar el DataFrame con las descripciones de los juegos
    df_juegos = pd.read_csv('app\data_for_recommender.csv')
    return model, df_juegos

# Inicializar Pinecone y conectar al índice
def inicializar_pinecone():
    pc = Pinecone(api_key='e2659d52-b976-4624-b8b8-8de36f8ea15a')  # Coloca tu API key
    # Nombre del índice
    index_name = "boardgames-recommendation"
    # Conectar al índice existente
    index = pc.Index(index_name)
    return index

# Función para recomendar juegos utilizando Pinecone con o sin valoración
def recomendar_juegos_pinecone(prompt, model, index, df_juegos, idioma, considerar_valoracion, top_n=5, w=0.7):
    # Traducir el prompt si está en español
    if idioma == "Español":
        translator = Translator()
        translated_prompt = translator.translate(prompt, src='es', dest='en').text
    else:
        translated_prompt = prompt

    # Generar el embedding del prompt traducido
    prompt_vector = model.infer_vector(translated_prompt.split())
    
    # Consultar Pinecone para obtener los juegos más similares
    result = index.query(vector=prompt_vector.tolist(), top_k=top_n * 2, include_values=False)

    # Obtener las recomendaciones
    recomendaciones = []
    for match in result['matches']:
        game_id = match['id']
        score = match['score']  # Similaridad del coseno devuelta por Pinecone

        # Buscar el juego en el DataFrame por su ID
        game_row = df_juegos[df_juegos['BGGId'] == int(game_id)]
        if not game_row.empty:
            nombre_juego = game_row['Name'].values[0]
            valoracion = game_row['Average_Rating'].values[0] if considerar_valoracion else 0  # Valoración solo si el usuario lo permite

            if considerar_valoracion and valoracion > 0:
                # Opción 1: Considerar la valoración, excluyendo los juegos con rating 0
                valoracion_normalizada = valoracion / 10  # Normalizamos la valoración
                similaridad_ponderada = (score * w) + (valoracion_normalizada * (1 - w))
                recomendaciones.append((nombre_juego, similaridad_ponderada, score, valoracion))
            elif not considerar_valoracion or valoracion == 0:
                # Opción 2: Solo usar la similaridad del coseno si no se considera la valoración o si el rating es 0
                recomendaciones.append((nombre_juego, score, score, valoracion))

    # Ordenar por similaridad (ponderada si se considera la valoración)
    recomendaciones.sort(key=lambda x: x[1], reverse=True)

    # Mostrar los resultados
    if idioma == "English":
        st.write("Recommended games:")
    else:
        st.write("Juegos recomendados:")

    for nombre_juego, similaridad_ponderada, score, valoracion in recomendaciones[:top_n]:
        if idioma == "English":
            if considerar_valoracion:
                st.write(f"{nombre_juego} (Similarity: {score * 100:.2f}%, Rating: {valoracion:.2f})")
            else:
                st.write(f"{nombre_juego} (Similarity: {score * 100:.2f}%)")
        else:
            if considerar_valoracion:
                st.write(f"{nombre_juego} (Similaridad: {score * 100:.2f}%, Valoración: {valoracion:.2f})")
            else:
                st.write(f"{nombre_juego} (Similaridad: {score * 100:.2f}%)")


# Inicializar la app de Streamlit
st.title("Board Game Recommender")

# Seleccionar idioma
idioma = st.selectbox("Choose your language / Elige tu idioma", ["English", "Español"])

# Inicializar Pinecone
index = inicializar_pinecone()

# Cargar el modelo y datos
model, df_juegos = cargar_modelo_y_datos()

# Opción para considerar la valoración
considerar_valoracion = st.checkbox("Considerar la valoración del juego" if idioma == "Español" else "Consider game rating")

# Pedir al usuario el prompt
if idioma == "English":
    prompt_usuario = st.text_input("Enter a prompt describing the kind of game you want:")
else:
    prompt_usuario = st.text_input("Ingresa una descripción del tipo de juego que buscas:")

# Botón para hacer la recomendación
if st.button("Recommend" if idioma == "English" else "Recomendar"):
    if prompt_usuario:
        recomendar_juegos_pinecone(prompt_usuario, model, index, df_juegos, idioma, considerar_valoracion)
    else:
        if idioma == "English":
            st.write("Please enter a prompt.")
        else:
            st.write("Por favor, ingresa una descripción.")
