import streamlit as st
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from sklearn.preprocessing import normalize
from pinecone import Pinecone, ServerlessSpec
from googletrans import Translator 
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Descargar stopwords si no están disponibles
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Cargar el modelo y los datos
@st.cache_resource
def cargar_modelo_y_datos():
    # Cargar el modelo Doc2Vec (asegúrate de guardar y cargar el modelo previamente entrenado)
    model = Doc2Vec.load("model/model_doc2vec.bin")
    
    # Cargar el DataFrame con las descripciones de los juegos
    df_juegos = pd.read_csv('data/cleaned/data_for_recommender.csv')
    return model, df_juegos

# Inicializar Pinecone y conectar al índice
def inicializar_pinecone():

    pc = Pinecone(api_key="e2659d52-b976-4624-b8b8-8de36f8ea15a")  
    # Nombre del índice
    index_name = "boardgames-recommendation"
    # Conectar al índice existente
    index = pc.Index(index_name)
    return index

# Función para preprocesar descripciones
def preprocesar_descripcion(texto):
    """
    Esta función tokeniza, elimina stopwords y lematiza el texto de entrada.
    
    Parámetros:
    - texto (str): Texto a preprocesar.
    
    Retorna:
    - str: Texto preprocesado.
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Eliminar entidades HTML y otros caracteres no deseados
    texto = re.sub(r'&#?\w+;', ' ', texto)  # Reemplazar entidades HTML
    texto = re.sub(r'\W+', ' ', texto)  # Eliminar caracteres no alfanuméricos
    texto = re.sub(r'\d+', '', texto)  # Eliminar números
    
    # Tokenización del texto
    tokens = word_tokenize(texto.lower())  # Convertir a minúsculas y tokenizar
    
    # Eliminar stopwords y lematizar los tokens
    tokens_limpios = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    
    # Unir los tokens preprocesados en un solo string
    return " ".join(tokens_limpios)

# Función para recomendar juegos utilizando Pinecone con o sin valoración
def recomendar_juegos_pinecone(prompt, model, index, df_juegos, idioma, considerar_valoracion, top_n=5, w=0.7):
    """
    Función que recomienda juegos usando Pinecone para la búsqueda de embeddings y permite considerar valoraciones.
    
    Parámetros:
    - prompt (str): Texto del usuario describiendo sus preferencias.
    - model (Doc2Vec): Modelo Doc2Vec entrenado.
    - index (pinecone.Index): Índice Pinecone donde están almacenados los embeddings.
    - df_juegos (DataFrame): DataFrame que contiene los datos de los juegos.
    - idioma (str): Idioma del prompt ('Español' o 'English').
    - considerar_valoracion (bool): Indica si se debe considerar la valoración de los juegos en la recomendación.
    - top_n (int): Número de recomendaciones a devolver.
    - w (float): Peso para la similaridad del coseno en la combinación ponderada.
    
    Retorna:
    - None (muestra los juegos recomendados en la salida estándar).
    """
    
    # Traducir el prompt si está en español
    if idioma == "Español":
        translator = Translator()
        translated_prompt = translator.translate(prompt, src='es', dest='en').text
    else:
        translated_prompt = prompt

    translated_prompt = preprocesar_descripcion(translated_prompt)

    # Generar el embedding del prompt traducido
    prompt_vector = model.infer_vector(translated_prompt.split())

    # Realizar la consulta a Pinecone para obtener los juegos más similares
    result = index.query(vector=normalize([prompt_vector])[0].tolist(), top_k=top_n * 2, include_values=False)

    # Obtener las recomendaciones
    recomendaciones = []
    for match in result['matches']:
        game_id = match['id']
        score = match['score']  # Similaridad del coseno devuelta por Pinecone (ya normalizada)

        # Buscar el juego en el DataFrame por su ID
        game_row = df_juegos[df_juegos['BGGId'] == int(game_id)]
        if not game_row.empty:
            nombre_juego = game_row['Name'].values[0]
            valoracion = game_row['Average_Rating'].values[0]  # Obtener la valoración del juego

            # Si se selecciona considerar la valoración y el juego tiene una valoración mayor a 0
            if considerar_valoracion:
                if valoracion > 0:  # Excluir juegos con valoración 0
                    valoracion_normalizada = valoracion / 10  # Normalizamos la valoración
                    similaridad_ponderada = (score * w) + (valoracion_normalizada * (1 - w))
                    recomendaciones.append((nombre_juego, similaridad_ponderada, score, valoracion))
            else:
                # Si no se considera la valoración, usar solo la similaridad del coseno
                recomendaciones.append((nombre_juego, score, score, valoracion))

    # Ordenar por similaridad ponderada o por la similaridad del coseno
    recomendaciones.sort(key=lambda x: x[1], reverse=True)

    # Mostrar los resultados en función del idioma
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
st.title("BoardGameMatcher")

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
