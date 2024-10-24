import requests
import xml.etree.ElementTree as ET
import pandas as pd
from time import sleep
import random

# Función para obtener detalles generales del juego junto con mecánicas y categorías
def obtener_detalles_completos(id_juego):
    """
    Esta función realiza una solicitud a la API de BoardGameGeek para obtener los detalles generales de un juego,
    incluyendo sus mecánicas y categorías, usando su ID.

    Parámetros:
    - id_juego (int): ID del juego en la base de datos de BoardGameGeek.

    Retorna:
    - dict: Un diccionario con los detalles completos del juego (si existe), incluyendo nombre, año de publicación,
            descripción, mecánicas, categorías, valoraciones, etc.
    - None: Si no se obtienen detalles del juego o si el ID es inválido.
    """
    url = f'https://www.boardgamegeek.com/xmlapi2/thing?id={id_juego}&stats=1'
    respuesta = requests.get(url)
    
    if respuesta.status_code == 200:
        xml_root = ET.fromstring(respuesta.content)
        
        # Verificar si el juego tiene nombre para determinar si es válido
        nombre_tag = xml_root.find('item/name')
        if nombre_tag is None:
            return None
        
        # Obtener detalles generales del juego
        nombre = nombre_tag.attrib['value']
        year_published = xml_root.find('item/yearpublished')
        year_published = year_published.attrib['value'] if year_published is not None else 'N/A'
        
        description = xml_root.find('item/description').text or ""
        description = description.replace('\n', ' ').replace('&quot;', '"').replace('&amp;', '&')
        
        min_players = xml_root.find('item/minplayers')
        min_players = min_players.attrib['value'] if min_players is not None else 'N/A'
        
        max_players = xml_root.find('item/maxplayers')
        max_players = max_players.attrib['value'] if max_players is not None else 'N/A'
        
        min_playtime = xml_root.find('item/minplaytime')
        min_playtime = min_playtime.attrib['value'] if min_playtime is not None else 'N/A'
        
        max_playtime = xml_root.find('item/maxplaytime')
        max_playtime = max_playtime.attrib['value'] if max_playtime is not None else 'N/A'
        
        # Valoraciones
        stats = xml_root.find('item/statistics/ratings')
        avg_rating = stats.find('average')
        avg_rating = avg_rating.attrib['value'] if avg_rating is not None else 'N/A'
        
        bayes_avg_rating = stats.find('bayesaverage')
        bayes_avg_rating = bayes_avg_rating.attrib['value'] if bayes_avg_rating is not None else 'N/A'
        
        num_ratings = stats.find('usersrated')
        num_ratings = num_ratings.attrib['value'] if num_ratings is not None else 'N/A'
        
        # Obtener mecánicas y categorías
        mecanicas = []
        categorias = []
        for link in xml_root.findall('item/link'):
            link_type = link.attrib.get('type')
            link_value = link.attrib.get('value')
            if link_type == 'boardgamemechanic':
                mecanicas.append(link_value)
            elif link_type == 'boardgamecategory':
                categorias.append(link_value)
        
        # Guardar todos los datos en un diccionario
        game = {
            'BGGId': id_juego,
            'Name': nombre,
            'Year_Published': year_published,
            'Description': description,
            'Min_Players': min_players,
            'Max_Players': max_players,
            'Min_Playtime': min_playtime,
            'Max_Playtime': max_playtime,
            'Average_Rating': avg_rating,
            'Bayesian_Average_Rating': bayes_avg_rating,
            'Number_of_Ratings': num_ratings,
            'Mechanics': mecanicas,
            'Categories': categorias
        }
        
        return game
    else:
        return None

# Función para obtener 1000 juegos válidos generando IDs aleatorios continuamente
def obtener_juegos_aleatorios(cantidad_deseada=1000, rango_min=1, rango_max=300000, pause_time=2):
    """
    Genera IDs aleatorios de juegos dentro de un rango dado y obtiene los detalles de esos juegos.
    Continúa hasta obtener la cantidad deseada de juegos válidos.

    Parámetros:
    - cantidad_deseada (int): Número de juegos válidos a obtener.
    - rango_min (int): Valor mínimo del ID para la generación aleatoria.
    - rango_max (int): Valor máximo del ID para la generación aleatoria.
    - pause_time (int): Pausa en segundos entre las solicitudes a la API para evitar sobrecarga.

    Retorna:
    - pd.DataFrame: Un DataFrame con los detalles de los juegos válidos obtenidos.
    """
    datos_juegos = []
    ids_generados = set()  # Usar un conjunto para evitar IDs duplicados
    
    while len(datos_juegos) < cantidad_deseada:
        # Generar un ID aleatorio dentro del rango
        id_juego = random.randint(rango_min, rango_max)
        
        # Evitar volver a generar el mismo ID
        if id_juego in ids_generados:
            continue
        
        # Intentar obtener los detalles del juego
        detalles_juego = obtener_detalles_completos(id_juego)
        if detalles_juego:
            datos_juegos.append(detalles_juego)  # Añadir a la lista si se obtienen datos completos
            print(f"Juego encontrado: {id_juego}, Total juegos obtenidos: {len(datos_juegos)}")
        
        # Añadir el ID a los generados para no repetir
        ids_generados.add(id_juego)
        
        # Pausa entre solicitudes para no saturar la API
        sleep(pause_time)
    
    # Convertir los datos recopilados en un DataFrame
    df_juegos = pd.DataFrame(datos_juegos)
    
    return df_juegos