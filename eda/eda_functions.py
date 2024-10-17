import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 1. Distribución de la Valoración Promedio
def analizar_distribucion_valoracion(df):
    plt.figure(figsize=(10,6))
    sns.histplot(df['Average_Rating'], bins=30, kde=True)
    plt.title('Distribución de la Valoración Promedio')
    plt.xlabel('Valoración Promedio')
    plt.ylabel('Frecuencia')
    plt.show()

# 2. Distribución de Juegos por Año de Publicación
def analizar_juegos_por_año(df):
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

# 3. Análisis de Jugadores y Tiempo de Juego
def analizar_jugadores_tiempo(df):
    # Distribución de la cantidad mínima de jugadores
    plt.figure(figsize=(10,6))
    sns.countplot(x='Min_Players', data=df)
    plt.title('Distribución de la Cantidad Mínima de Jugadores')
    plt.xlabel('Jugadores Mínimos')
    plt.ylabel('Cantidad de Juegos')
    plt.show()

    # Distribución de tiempo mínimo de juego
    df['Min_Playtime'] = pd.to_numeric(df['Min_Playtime'], errors='coerce')

    plt.figure(figsize=(10,6))
    sns.histplot(df['Min_Playtime'], bins=20, kde=True)
    plt.title('Distribución del Tiempo Mínimo de Juego')
    plt.xlabel('Minutos')
    plt.ylabel('Cantidad de Juegos')
    plt.show()

# 4. Análisis de Mecánicas y Categorías Populares
def analizar_mecanicas_categorias(df):
    # Separar las mecánicas y contar las ocurrencias
    mecanicas_series = df['Mechanics'].explode()
    mecanicas_populares = mecanicas_series.value_counts()

    # Graficar las mecánicas más populares
    plt.figure(figsize=(12,6))
    sns.barplot(x=mecanicas_populares.values[:10], y=mecanicas_populares.index[:10])
    plt.title('Top 10 Mecánicas Más Populares')
    plt.xlabel('Cantidad de Juegos')
    plt.ylabel('Mecánica')
    plt.show()

    # Separar y contar las categorías
    categorias_series = df['Categories'].explode()
    categorias_populares = categorias_series.value_counts()

    plt.figure(figsize=(12,6))
    sns.barplot(x=categorias_populares.values[:10], y=categorias_populares.index[:10])
    plt.title('Top 10 Categorías Más Populares')
    plt.xlabel('Cantidad de Juegos')
    plt.ylabel('Categoría')
    plt.show()

# 5. Correlación entre Valoraciones y Otras Variables
def analizar_correlaciones(df):
    df['Number_of_Ratings'] = pd.to_numeric(df['Number_of_Ratings'], errors='coerce')
    df['Average_Rating'] = pd.to_numeric(df['Average_Rating'], errors='coerce')

    # Calcular la correlación entre valoraciones y otras variables
    correlation_matrix = df[['Average_Rating', 'Min_Players', 'Max_Players', 'Min_Playtime', 'Max_Playtime', 'Number_of_Ratings']].corr()

    # Mostrar un heatmap de la correlación
    plt.figure(figsize=(10,6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlación entre Valoraciones y Otras Variables')
    plt.show()

# 6. Juegos Mejor Valorados por Mecánica o Categoría
def analizar_mejores_por_mecanica_categoria(df):
    # Explode las mecánicas y calcular la valoración promedio por mecánica
    df_exploded_mecanicas = df.explode('Mechanics')
    rating_por_mecanica = df_exploded_mecanicas.groupby('Mechanics')['Average_Rating'].mean().sort_values(ascending=False)

    # Mostrar las mecánicas con mejor valoración
    plt.figure(figsize=(12,6))
    sns.barplot(x=rating_por_mecanica.values[:10], y=rating_por_mecanica.index[:10])
    plt.title('Mecánicas Mejor Valoradas')
    plt.xlabel('Valoración Promedio')
    plt.ylabel('Mecánica')
    plt.show()

    # Hacer lo mismo para categorías
    df_exploded_categorias = df.explode('Categories')
    rating_por_categoria = df_exploded_categorias.groupby('Categories')['Average_Rating'].mean().sort_values(ascending=False)

    plt.figure(figsize=(12,6))
    sns.barplot(x=rating_por_categoria.values[:10], y=rating_por_categoria.index[:10])
    plt.title('Categorías Mejor Valoradas')
    plt.xlabel('Valoración Promedio')
    plt.ylabel('Categoría')
    plt.show()
