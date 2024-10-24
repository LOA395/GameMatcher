# recommender.py

import re
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from gensim.models.doc2vec import Doc2Vec
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import yaml

# Función para cargar la configuración desde config.yaml
def cargar_configuracion():
    with open("..\config.yaml", "r") as archivo_config:
        config = yaml.safe_load(archivo_config)
    return config


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

# Función para recomendar juegos usando Doc2Vec
def recomendar_juegos_doc2vec(prompt, df_descripciones, model, top_n=5):
    """
    Recomienda juegos en base a la similaridad del coseno entre el prompt y los juegos.
    
    Parámetros:
    - prompt (str): Texto de entrada del usuario describiendo sus preferencias.
    - df_descripciones (DataFrame): DataFrame que contiene las descripciones de los juegos.
    - model (Doc2Vec): Modelo Doc2Vec entrenado.
    - top_n (int): Número de recomendaciones a devolver.
    
    Retorna:
    - None (imprime los juegos recomendados en la salida estándar).
    """
    # Preprocesar el prompt y vectorizarlo usando Doc2Vec
    prompt = preprocesar_descripcion(prompt)
    prompt_vector = model.infer_vector(prompt.split())
    
    # Obtener los vectores de todos los juegos y normalizarlos
    vectors = [model.dv[i] for i in range(len(df_descripciones))]
    vectors_normalized = normalize(vectors)
    
    # Normalizar el vector del prompt
    prompt_vector_normalized = normalize([prompt_vector])
    
    # Calcular la similaridad del coseno entre el prompt y los juegos normalizados
    cosine_similarities = cosine_similarity(prompt_vector_normalized, vectors_normalized).flatten()

    # Obtener los índices de los juegos con mayor similaridad
    indices_recomendados = cosine_similarities.argsort()[-top_n:][::-1]

    # Mostrar los juegos recomendados
    print("Juegos recomendados:")
    for index in indices_recomendados:
        print(f"{df_descripciones.iloc[index]['Name']} (Similaridad: {cosine_similarities[index]:.4f})")


# Función para recomendar juegos con valoraciones ponderadas
def recomendar_juegos_doc2vec_con_valoraciones(prompt, df_descripciones, model, top_n=5, umbral_similaridad=0.1, w=0.7):
    """
    Recomienda juegos considerando tanto la similaridad del coseno como las valoraciones de los juegos.
    
    Parámetros:
    - prompt (str): Texto de entrada del usuario describiendo sus preferencias.
    - df_descripciones (DataFrame): DataFrame que contiene las descripciones de los juegos.
    - model (Doc2Vec): Modelo Doc2Vec entrenado.
    - top_n (int): Número de recomendaciones a devolver.
    - umbral_similaridad (float): Umbral de similaridad mínima para considerar una recomendación.
    - w (float): Peso para la similaridad del coseno en la combinación ponderada.
    
    Retorna:
    - None (imprime los juegos recomendados en la salida estándar).
    """
    # Preprocesar el prompt y vectorizarlo usando Doc2Vec
    prompt = preprocesar_descripcion(prompt)
    prompt_vector = model.infer_vector(prompt.split())
    
    # Normalizar el vector del prompt
    prompt_vector_normalized = normalize([prompt_vector])[0]
    
    # Obtener los vectores de todos los juegos y normalizarlos
    vectors = [model.dv[i] for i in range(len(df_descripciones))]
    vectors_normalized = normalize(vectors)

    # Calcular la similaridad del coseno entre el prompt y los juegos
    cosine_similarities = cosine_similarity([prompt_vector_normalized], vectors_normalized).flatten()

    # Asegurarse de que las valoraciones coincidan con los juegos vectorizados
    valoraciones = df_descripciones.loc[df_descripciones.index.isin(df_descripciones.index), 'Average_Rating'].fillna(0).values
    valoraciones_normalizadas = valoraciones / 10  # Normalizar las valoraciones a una escala de 0 a 1

    # Asegurarse de que las dimensiones coincidan
    if len(cosine_similarities) != len(valoraciones):
        min_len = min(len(cosine_similarities), len(valoraciones))
        cosine_similarities = cosine_similarities[:min_len]
        valoraciones_normalizadas = valoraciones_normalizadas[:min_len]

    # Crear una métrica ponderada que priorice más la similaridad del coseno
    similaridad_ponderada = (cosine_similarities * w) + (valoraciones_normalizadas * (1 - w))

    # Filtrar recomendaciones con similaridad negativa o muy baja
    indices_validos = [i for i, sim in enumerate(cosine_similarities) if sim > umbral_similaridad]

    # Ordenar los juegos con mayor similaridad ponderada
    indices_recomendados = sorted(indices_validos, key=lambda i: similaridad_ponderada[i], reverse=True)[:top_n]

    # Mostrar los juegos recomendados
    print("Juegos recomendados (priorizando similaridad del coseno):")
    for index in indices_recomendados:
        print(f"{df_descripciones.iloc[index]['Name']} (Similaridad ponderada: {similaridad_ponderada[index]:.4f}, "
              f"Similaridad del coseno: {cosine_similarities[index]:.4f}, "
              f"Valoración: {valoraciones[index]:.2f})")


# Función para recomendar juegos con Pinecone
def recomendar_juegos_pinecone(prompt, model, index, df_juegos, top_n=5):
    """
    Recomienda juegos en base a la similaridad del coseno entre el prompt y los juegos almacenados en Pinecone.
    
    Parámetros:
    - prompt (str): Texto de entrada del usuario describiendo sus preferencias.
    - model (Doc2Vec): Modelo Doc2Vec entrenado.
    - index (Pinecone Index): Índice Pinecone conectado.
    - df_juegos (DataFrame): DataFrame que contiene los datos de los juegos.
    - top_n (int): Número de recomendaciones a devolver.
    
    Retorna:
    - None (imprime los juegos recomendados en la salida estándar).
    """
    # Preprocesar el prompt y vectorizarlo usando Doc2Vec
    prompt = preprocesar_descripcion(prompt)
    prompt_vector = model.infer_vector(prompt.split())
    prompt_vector_normalized = normalize([prompt_vector])[0]  # Normalizamos el vector
    
    # Consultar el índice de Pinecone
    result = index.query(vector=prompt_vector_normalized.tolist(), top_k=top_n, include_values=False)
    
    # Obtener los juegos recomendados
    recomendaciones = []
    for match in result['matches']:
        game_id = match['id']
        score = match['score']  # Similaridad del coseno devuelta por Pinecone

        # Buscar el juego en el DataFrame
        game_row = df_juegos[df_juegos['BGGId'] == int(game_id)]
        if not game_row.empty:
            recomendaciones.append((game_row['Name'].values[0], score))

    # Mostrar los resultados
    print("Juegos recomendados:")
    for rec in recomendaciones:
        print(f"{rec[0]} (Similaridad: {rec[1]:.4f})")

# Función para recomendar juegos con valoraciones ponderadas con Pinecone
def recomendar_juegos_pinecone_con_valoraciones(prompt, model, index, df_juegos, top_n=5, w=0.7):
    """
    Recomienda juegos considerando tanto la similaridad del coseno (desde Pinecone) como las valoraciones de los juegos.
    
    Parámetros:
    - prompt (str): Texto de entrada del usuario describiendo sus preferencias.
    - model (Doc2Vec): Modelo Doc2Vec entrenado.
    - index (Pinecone Index): Índice Pinecone conectado.
    - df_juegos (DataFrame): DataFrame que contiene los datos de los juegos.
    - top_n (int): Número de recomendaciones a devolver.
    - w (float): Peso para la similaridad del coseno en la combinación ponderada.
    
    Retorna:
    - None (imprime los juegos recomendados en la salida estándar).
    """
    # Preprocesar el prompt y vectorizarlo usando Doc2Vec
    prompt = preprocesar_descripcion(prompt)
    prompt_vector = model.infer_vector(prompt.split())
    prompt_vector_normalized = normalize([prompt_vector])[0]  # Normalizamos el vector
    
    # Consultar el índice de Pinecone
    result = index.query(vector=prompt_vector_normalized.tolist(), top_k=top_n * 2, include_values=False)

    recomendaciones = []
    for match in result['matches']:
        game_id = match['id']
        score = match['score']  # Similaridad del coseno devuelta por Pinecone

        # Buscar el juego en el DataFrame por su ID
        game_row = df_juegos[df_juegos['BGGId'] == int(game_id)]
        if not game_row.empty:
            valoracion = game_row['Average_Rating'].values[0]
            if valoracion > 0:  # Filtrar juegos con valoración mayor que 0
                valoracion_normalizada = valoracion / 10  # Normalizamos la valoración
                similaridad_ponderada = (score * w) + (valoracion_normalizada * (1 - w))
                recomendaciones.append((game_row['Name'].values[0], similaridad_ponderada, score, valoracion))

    # Ordenar por similaridad ponderada
    recomendaciones.sort(key=lambda x: x[1], reverse=True)

    # Mostrar los resultados
    print("Juegos recomendados (priorizando valoraciones):")
    for rec in recomendaciones[:top_n]:
        print(f"{rec[0]} (Similaridad ponderada: {rec[1]:.4f}, Similaridad: {rec[2]:.4f}, Valoración: {rec[3]:.2f})")

