from fastapi import FastAPI
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse


#Leer Archivos
df_items = pd.read_parquet('data_items.parquet')
df_reviews = pd.read_parquet('data_reviews.parquet')
df_games = pd.read_parquet('data_games_explode.parquet')
merged_df = pd.read_parquet('items_merged_games.parquet')

df_items['playtime_forever'] = (df_items['playtime_forever'] / 60).round(2)


app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Bienvenido a esta Api...dirigirse a /docs"}




@app.get("/PlayTimeGenre/{genero}")
def playTimeGenre( genero : str ):
    genero_especificado = genero.lower()
    #1. Realiza un merge entre las dos tablas utilizando la columna 'item_id' como clave de unión
    
    
    # 2. Filtra las filas por el género especificado (por ejemplo, 'Género X')
    filtro_por_genero = merged_df[merged_df['genres'] == genero_especificado]

    if filtro_por_genero.empty:
            error_message = {"error": "No hay reseñas para el genero proporcionado."}
            return JSONResponse(status_code=404, content=error_message)
    
    # 3. Agrupa los datos por año de lanzamiento y suma las horas jugadas en cada año
    grupo_por_anio = filtro_por_genero.groupby(filtro_por_genero['release_date'].dt.year)['playtime_forever'].sum()
    # 4. Encuentra el año con la suma máxima de horas jugadas
    anio_con_mas_horas_jugadas =int( grupo_por_anio.idxmax())
    # Crear el resultado como un diccionario
    resultado = {"Año de lanzamiento con más horas jugadas para " + genero_especificado: anio_con_mas_horas_jugadas}
    return resultado
 
@app.get("/UserForGenre/{genero}")
def UserForGenre( genero : str ):
  
    # Paso 3: Calcular la suma de horas jugadas por usuario y género
    horas_por_usuario = merged_df.groupby(['user_id', 'genres'])['playtime_forever'].sum().reset_index()
    
    # Paso 4: Encontrar el usuario con más horas jugadas para ese género
    usuario_max_horas = horas_por_usuario[horas_por_usuario['genres'] == genero].nlargest(1, 'playtime_forever')
    
    if usuario_max_horas.empty:
            error_message = {"error": "No hay reseñas para el genero proporcionado."}
            return JSONResponse(status_code=404, content=error_message)
   
    # Paso 5: Calcular la acumulación de horas jugadas por año para ese usuario y género
    horas_por_anio = merged_df[merged_df['user_id'] == usuario_max_horas.iloc[0]['user_id']]
    horas_por_anio['release_date'] = pd.to_datetime(horas_por_anio['release_date'])
    horas_por_anio['anio'] = horas_por_anio['release_date'].dt.year
    horas_por_anio = horas_por_anio.groupby('anio')['playtime_forever'].sum().round(2).reset_index()

    # Formatear el resultado
    resultado = {
        f'Usuario con más horas jugadas para {genero} ': usuario_max_horas.iloc[0]['user_id'],
        "Horas jugadas por año": [{"Año": int(row['anio']), "Horas": row['playtime_forever']} for _, row in horas_por_anio.iterrows()]
    }

    return resultado


@app.get("/UsersRecommend/{anio}")
def UsersRecommend(anio : int):
    # Paso 1: Filtrar las reseñas del año deseado
    reseñas_anio = df_reviews[df_reviews['posted'].dt.year == anio]
    if reseñas_anio.empty:
            error_message = {"error": "No hay reseñas para el año proporcionado."}
            return JSONResponse(status_code=404, content=error_message)
    # Paso 2: Filtrar las reseñas con recomendaciones positivas o neutrales
    reseñas_positivas_neutrales = reseñas_anio[reseñas_anio['recommend'] == True]
    reseñas_positivas_neutrales = reseñas_positivas_neutrales[reseñas_positivas_neutrales['sentiment_analysis'].isin([1, 2])]
    # Paso 3: Calcular el conteo de reseñas por juego
    conteo_reseñas_por_juego = reseñas_positivas_neutrales.groupby('item_id').size().reset_index(name='conteo')

    # Paso 4: Ordenar los juegos por conteo de reseñas en orden descendente
    juegos_mas_recomendados = conteo_reseñas_por_juego.sort_values(by='conteo', ascending=False)

    # Paso 5: Tomar los 3 juegos más recomendados (puedes ajustar esto según tu necesidad)
    top_3_juegos = juegos_mas_recomendados.head(3)
    top_3_juegos.reset_index(inplace = True)

    resultado = [{"Puesto {}: ".format(i + 1): juego['item_id']} for i, juego in top_3_juegos.iterrows()]

    return resultado



@app.get("/UsersNotRecommend/{anio}")
def UsersNotRecommend(anio : int):
    # Paso 1: Filtrar las reseñas del año deseado
    reseñas_anio = df_reviews[df_reviews['posted'].dt.year == anio]
    if reseñas_anio.empty:
            error_message = {"error": "No hay reseñas para el año proporcionado."}
            return JSONResponse(status_code=404, content=error_message)
    # Paso 2: Filtrar las reseñas con recomendaciones positivas o neutrales
    reseñas_negativas= reseñas_anio[reseñas_anio['recommend'] == False]
    reseñas_negativas = reseñas_negativas[reseñas_negativas['sentiment_analysis'] == 0]
    # Paso 3: Calcular el conteo de reseñas por juego
    conteo_reseñas_por_juego = reseñas_negativas.groupby('item_id').size().reset_index(name='conteo')

    # Paso 4: Ordenar los juegos por conteo de reseñas en orden descendente
    juegos_menos_recomendados = conteo_reseñas_por_juego.sort_values(by='conteo', ascending=False)

    # Paso 5: Tomar los 3 juegos más recomendados (puedes ajustar esto según tu necesidad)
    top_3_juegos = juegos_menos_recomendados.head(3)
    top_3_juegos.reset_index(inplace = True)

    # Formatear el resultado como una lista de diccionarios
    resultado = [{"Puesto {}: ".format(i + 1): juego['item_id']} for i, juego in top_3_juegos.iterrows()]

    return resultado


@app.get("/sentiment_analysis/{anio}")
def sentiment_analysis(anio : int ):
 
        # Filtra las reseñas por el año de lanzamiento
        reseñas_por_anio = df_reviews[df_reviews['posted'].dt.year == anio]
        if reseñas_por_anio.empty:
            error_message = {"error": "No hay reseñas para el año proporcionado."}
            return JSONResponse(status_code=404, content=error_message)
        # Calcula la cantidad de reseñas por categoría de sentimiento
        conteo_sentimientos = reseñas_por_anio['sentiment_analysis'].value_counts().to_dict()

        # Mapea los valores numéricos a etiquetas
        resultado = {
            'Negative': conteo_sentimientos.get(0, 0),
            'Neutral': conteo_sentimientos.get(1, 0),
            'Positive': conteo_sentimientos.get(2, 0)
        }

        return resultado


#Achicamos una y otra vez hasta que funcione en render  
df_reviews_mix = df_reviews.head(5000)
df_reviews_mix = df_reviews_mix.sample(frac=1, random_state=42) 

 # Crear una matriz de interacciones
user_item_matrix = df_reviews_mix.pivot_table(index='user_id', columns='item_id', values='sentiment_analysis').fillna(0)

# Calcular la similitud del coseno entre usuarios
user_similarity = cosine_similarity(user_item_matrix)

@app.get("/recomendacion_usuario/{id_usuario}")
def recomendacion_usuario(id_usuario : str):
    if id_usuario in user_item_matrix.index:
        # Obtener la fila correspondiente al usuario 
        user_vector = user_item_matrix.loc[id_usuario].values.reshape(1, -1)
        
        # Calcular la similitud entre el usuario ingresado y todos los demás usuarios
        similarities = cosine_similarity(user_vector, user_item_matrix)
        
        # Obtener los juegos recomendados en función de los usuarios similares
        user_reviews = user_item_matrix.loc[id_usuario]
        similar_users = df_reviews['user_id'][similarities.argsort()[0][-6:-1]]
        recommended_items = user_item_matrix.loc[similar_users].mean(axis=0).sort_values(ascending=False)
        
        # Filtrar los juegos que el usuario ya ha jugado
        recommended_items = recommended_items[recommended_items.index.isin(user_reviews[user_reviews == 0].index)]
        
        return recommended_items.index.tolist()[:5]
    else:
        # El usuario no está en user_item_matrix, devuelve los índices disponibles y un mensaje
        available_users = user_item_matrix.index.tolist()
        message = "El usuario no está en la base de datos. Pruebe con algunos de estos usuarios: " + ", ".join(available_users[:15])
        return message