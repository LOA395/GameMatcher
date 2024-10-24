import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px


# Función para exploración de los datos
def df_exploration(df):
    """
    Explora el DataFrame con información básica.
    
    Parámetros:
    - df: DataFrame a analizar.
    
    Salida:
    - Muestra tipos de datos, valores duplicados, valores nulos y únicos.
    """
    # Revisión básica de los datos
    print(df.info())  # Para revisar los tipos de datos
    print(f"\nValores duplicados: {df.duplicated().sum()}")  # Revisa si hay duplicados en los datasets
    print(f"\nValores nulos: \n{df.isnull().sum()}") # Revisar valores nulos
    print (f"\nValores unicos: \n{df.nunique()}") # Revisar valores unicos
    return df

# Expandir las columnas de mecánicas y categorías a formato binario
def expandir_columnas_binarias(df, columna):
    """
    Expande una columna de listas en varias columnas binarias, manteniendo la ID del juego.
    
    Parámetros:
    - df: DataFrame con los datos.
    - columna: Nombre de la columna a expandir.
    
    Salida:
    - DataFrame con la columna expandida a formato binario (0 o 1).
    """
    # Mantener la columna 'BGGId' y expandir la columna binaria
    df_expanded = df[['BGGId']].join(
        df[columna].str.strip('[]').str.replace("'", "").str.split(', ').explode().str.get_dummies().groupby(level=0).sum()
    )
    
    # Asegurar que los valores sean binarios (0 o 1), sin afectar la columna 'BGGId'
    df_expanded.loc[:, df_expanded.columns != 'BGGId'] = df_expanded.loc[:, df_expanded.columns != 'BGGId'].clip(upper=1)
    
    return df_expanded


# Análisis univariable de variables numéricas
def analizar_variable_numerica(df, columna):
    """
    Realiza un análisis de una variable numérica mediante gráficos y estadísticas descriptivas.
    
    Parámetros:
    - df: DataFrame a analizar.
    - columna: Nombre de la columna numérica.
    
    Salida:
    - Histograma, boxplot y descripción estadística de la columna.
    """
    plt.figure(figsize=(10,6))
    
    # Histograma
    sns.histplot(df[columna], bins=30, kde=True)
    plt.title(f'Distribución de {columna}')
    plt.xlabel(columna)
    plt.ylabel('Frecuencia')
    plt.show()
    
    # Boxplot
    plt.figure(figsize=(10,6))
    sns.boxplot(x=df[columna])
    plt.title(f'Boxplot de {columna}')
    plt.show()

    # Descripción estadística
    descripcion = df[columna].describe()
    print(f'Descripción estadística de {columna}:\n{descripcion}\n')

# Análisis univariable de variables categóricas
def analizar_variable_categorica(df, columna):
    """
    Realiza un análisis de una variable categórica mediante gráficos de barras.
    
    Parámetros:
    - df: DataFrame a analizar.
    - columna: Nombre de la columna categórica.
    
    Salida:
    - Gráfico de barras con frecuencias y la distribución de las categorías.
    """
    plt.figure(figsize=(10,6))
    
    # Gráfico de barras con conteo de la columna categórica
    sns.countplot(y=columna, data=df, order=df[columna].value_counts().index)
    plt.title(f'Frecuencia de {columna}')
    plt.xlabel('Frecuencia')
    plt.ylabel(columna)
    plt.show()

    # Descripción de la frecuencia de cada categoría
    frecuencia = df[columna].value_counts()
    print(f'Frecuencia de {columna}:\n{frecuencia}\n')

# Análisis univariable de variables textuales
def analizar_variable_textual(df, columna):
    """
    Analiza la longitud de los textos de una columna y genera gráficos de la distribución.
    
    Parámetros:
    - df: DataFrame a analizar.
    - columna: Nombre de la columna de texto.
    
    Salida:
    - Histograma de la longitud y estadísticas descriptivas de los textos.
    """
    # Longitud de los textos
    df[f'{columna}_longitud'] = df[columna].str.len()
    
    plt.figure(figsize=(10,6))
    sns.histplot(df[f'{columna}_longitud'], bins=30, kde=True)
    plt.title(f'Longitud de los textos en {columna}')
    plt.xlabel('Longitud de Texto')
    plt.ylabel('Frecuencia')
    plt.show()

    # Descripción estadística de la longitud de los textos
    descripcion_longitud = df[f'{columna}_longitud'].describe()
    print(f'Descripción estadística de la longitud de los textos en {columna}:\n{descripcion_longitud}\n')


# Distribución de Juegos por Año de Publicación
def analizar_juegos_por_año(df):
    """
    Grafica la distribución de los juegos según su año de publicación.
    
    Parámetros:
    - df: DataFrame con los juegos y su año de publicación.
    
    Salida:
    - Gráfico de líneas con la cantidad de juegos publicados por año.
    """
    df['Year_Published'] = pd.to_numeric(df['Year_Published'], errors='coerce')

    # Contar la cantidad de juegos por año
    juegos_por_año = df['Year_Published'].value_counts().sort_index()

    # Graficar la cantidad de juegos por año
    plt.figure(figsize=(12,6))
    sns.lineplot(x=juegos_por_año.index, y=juegos_por_año.values)
    plt.title('Cantidad de Juegos Publicados por Año')
    plt.xlabel('Año')
    plt.ylabel('Número de Juegos')
    plt.show()

# Análisis de Mecánicas Populares
def analizar_mecanicas_populares(df_mecanicas):
    """
    Muestra un gráfico de barras con las 10 mecánicas más populares en los juegos.
    
    Parámetros:
    - df_mecanicas: DataFrame con las mecánicas en formato binario.
    
    Salida:
    - Gráfico de barras con las mecánicas más populares.
    """
    mecánicas_populares = df_mecanicas.drop('BGGId', axis=1).sum().sort_values(ascending=False).head(10)
    
    plt.figure(figsize=(12,6))
    sns.barplot(x=mecánicas_populares.values, y=mecánicas_populares.index)
    plt.title('Top 10 Mecánicas Más Populares')
    plt.xlabel('Cantidad de Juegos')
    plt.ylabel('Mecánica')
    plt.show()

# Categorías Más Populares
def analizar_categorias_populares(df_categorias):
    """
    Muestra un gráfico de barras con las 10 categorías más populares en los juegos.
    
    Parámetros:
    - df_categorias: DataFrame con las categorías en formato binario.
    
    Salida:
    - Gráfico de barras con las categorías más populares.
    """
    categorias_populares = df_categorias.drop('BGGId', axis=1).sum().sort_values(ascending=False).head(10)
    
    plt.figure(figsize=(12,6))
    sns.barplot(x=categorias_populares.values, y=categorias_populares.index)
    plt.title('Top 10 Categorías Más Populares')
    plt.xlabel('Cantidad de Juegos')
    plt.ylabel('Categoría')
    plt.show()

# Correlación entre Valoraciones y Otras Variables
def analizar_correlaciones(df):
    """
    Muestra la matriz de correlación entre varias variables numéricas.
    
    Parámetros:
    - df: DataFrame con las variables numéricas a analizar.
    
    Salida:
    - Heatmap de correlación entre las variables seleccionadas.
    """
    # Calcular la correlación entre valoraciones y otras variables
    correlation_matrix = df[['Bayesian_Average_Rating', 'Average_Rating', 'Min_Players', 'Max_Players', 'Min_Playtime', 'Max_Playtime', 'Number_of_Ratings']].corr()

    # Mostrar un heatmap de la correlación
    plt.figure(figsize=(10,6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlación entre Valoraciones y Otras Variables')
    plt.show()

# Analizar las Mecánicas y Categorías Mejor Valoradas
def analizar_mejor_valoradas(df_juegos, df_binario, tipo):
    """
    Función para analizar las mejores valoradas para mecánicas o categorías, considerando solo valoraciones > 0.
    
    Parámetros:
    - df_juegos: DataFrame con los juegos y sus valoraciones.
    - df_binario: DataFrame binario (de mecánicas o categorías).
    - tipo: String que indica si es 'mecánicas' o 'categorías'.
    
    Salida:
    - Gráfico de barras y descripción de las 10 mejores mecánicas o categorías valoradas.
    """
    # Filtrar juegos con valoraciones mayores a 0
    df_juegos_filtrado = df_juegos[df_juegos['Average_Rating'] > 0]
    
    # Unir el DataFrame de juegos filtrado con el DataFrame binario (mecánicas o categorías)
    df_completo = df_juegos_filtrado[['BGGId', 'Average_Rating']].merge(df_binario, on='BGGId')
    
    # Calcular la media de las valoraciones para cada mecánica o categoría
    valoracion = {}
    for columna in df_binario.columns[1:]:  # Iterar sobre todas las mecánicas o categorías
        juegos_con_columna = df_completo[df_completo[columna] == 1]
        if len(juegos_con_columna) > 0:
            valoracion[columna] = juegos_con_columna['Average_Rating'].mean()
        else:
            valoracion[columna] = 0  # Si no hay juegos con esa mecánica/categoría, asignamos 0

    # Convertir el diccionario a una serie para ordenar y graficar
    valoracion_series = pd.Series(valoracion).sort_values(ascending=False)
    
    # Graficar las 10 mejores valoradas (ya sean mecánicas o categorías)
    plt.figure(figsize=(12,6))
    sns.barplot(x=valoracion_series.values[:10], y=valoracion_series.index[:10])
    plt.title(f'Top 10 {tipo.capitalize()} Mejor Valoradas')
    plt.xlabel('Valoración Promedio')
    plt.ylabel(tipo.capitalize())
    plt.show()

    # Mostrar los resultados en texto
    print(f'Top 10 {tipo.capitalize()} Mejor Valoradas:\n{valoracion_series.head(10)}\n')



