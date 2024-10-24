# BoardGameMatcher: Encuentra tu Juego Perfecto

BoardGameMatcher es una aplicación diseñada para ayudar a los usuarios a encontrar juegos de mesa adaptados a sus preferencias. Utiliza técnicas avanzadas de Procesamiento del Lenguaje Natural (NLP) y Machine Learning para ofrecer recomendaciones personalizadas basadas en las descripciones y características de los juegos.

## Objetivos del Proyecto

1. **Recomendador Inteligente**  
   Crear un sistema avanzado de recomendación de juegos de mesa utilizando inteligencia artificial (IA).

2. **Procesamiento del Lenguaje Natural**  
   Implementar técnicas de NLP para analizar descripciones, mecánicas y categorías de los juegos de mesa.

3. **Experiencia Personalizada**  
   Proveer recomendaciones adaptadas a las preferencias de cada usuario, incluyendo la opción de tener en cuenta las valoraciones de los juegos.

## Estructura del Proyecto

### 1. Extracción de Datos
Los datos de los juegos se obtienen mediante la **API de BoardGameGeek (BGG)**. De esta manera, se recolectaron datos de **10,000 juegos**, incluyendo información clave como:
- Descripción
- Mecánicas
- Categorías
- Valoraciones
- Año de publicación
- Número de jugadores y tiempo de partida.

Se desarrolló un sistema para manejar errores y asegurar la validez de los juegos extraídos, ya que la API de BGG puede presentar limitaciones de velocidad y disponibilidad.

### 2. Preprocesamiento de Datos
Antes de entrenar el modelo, se aplicaron varias técnicas de limpieza y transformación a los datos:
- **Limpieza de Texto**: Eliminación de caracteres especiales y normalización.
- **Lematización**: Reducción de palabras a su forma base.
- **Eliminación de Stopwords**: Remover palabras comunes que no aportan valor semántico.
- **Fusión de Descripciones**: Se integraron las mecánicas y categorías de los juegos con las descripciones.

### 3. Vectorización y Modelo de Recomendación
Utilizamos **Doc2Vec** para vectorizar las descripciones y características de los juegos, transformándolas en vectores numéricos de alta dimensionalidad. Este modelo captura la semántica de los juegos para encontrar similitudes entre ellos.

#### Normalización de Vectores
Los vectores se normalizan para mejorar la precisión en la búsqueda.

#### Integración con Pinecone
Se emplea **Pinecone**, una base de datos para embeddings, para almacenar y buscar los vectores de los juegos, lo que permite una búsqueda eficiente y escalable.

### 4. Aplicación en Streamlit
El proyecto incluye una interfaz intuitiva construida en **Streamlit**, donde el usuario puede:
- Describir sus preferencias en lenguaje natural.
- Elegir si desea tener en cuenta las valoraciones de los juegos.
- Obtener recomendaciones personalizadas basadas en la similaridad del coseno entre los juegos y las preferencias del usuario.

### 5. Análisis de Datos
Se llevó a cabo un análisis exploratorio de los datos, incluyendo visualizaciones en **Python** y **Power BI**, para entender mejor las características de los juegos.

![Demo del Power BI](https://s2.ezgif.com/tmp/ezgif-2-9987c947f5.gif)


## Requisitos

- **Python 3.9**
- **numpy**
- **pandas**
- **matplotlib**
- **seaborn**
- **plotly**
- **scikit-learn**
- **nltk**
- **gensim**
- **streamlit**
- **pinecone-client**
- **googletrans**

## Ejecución de la Aplicación

Para ejecutar la aplicación en tu entorno local:

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/LOA395/GameMatcher.git
   cd GameMatcher
2. Instalar las dependencias:
    ```bash
    pip install -r requirements.txt
4. Ejecutar la aplicación de Streamlit:
   ```bash
   streamlit run app.py
6. Acceder a la aplicación en tu navegador en: http://localhost:8501

## Resources

- [Presentación]()
- [Power BI](https://drive.google.com/file/d/1KZK5Wnq67g3Et1P-4A0IImltThUeYFax/view?usp=sharing)
- [Streamlit](https://boardgamematcher.streamlit.app/)

## Autores:
  - `Laura Ortiz Alameda` - [Linkedin](https://www.linkedin.com/in/laura-ortiz-alameda/)
