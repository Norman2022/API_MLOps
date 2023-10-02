from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse


#Leer Archivos
df_items = pd.read_parquet('data_items.parquet')
df_reviews = pd.read_parquet('data_reviews.parquet')
df_games = pd.read_parquet('data_games_explode.parquet')

df_items['playtime_forever'] = (df_items['playtime_forever'] / 60).round(2)


app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Bienvenido a la Api de Steam"}


@app.get("/PlayTimeGenre/{genero}")
def playTimeGenre( genero : str ):
    genero_especificado = genero.lower()
    #1. Realiza un merge entre las dos tablas utilizando la columna 'item_id' como clave de unión
    df_games_copy  = df_games[df_games['genres'] == genero_especificado]
    merged_df = pd.merge(df_games_copy, df_items, on='item_id', how='inner')
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

    # Paso 1: Filtrar los juegos del género deseado     
    juegos_genero = df_games[df_games['genres'] == genero.lower()]
   
    # Paso 2: Unir df_items y juegos_genero en función de item_id
    df_merged = pd.merge(df_items, juegos_genero, on='item_id')

    # Paso 3: Calcular la suma de horas jugadas por usuario y género
    horas_por_usuario = df_merged.groupby(['user_id', 'genres'])['playtime_forever'].sum().reset_index()
    
    # Paso 4: Encontrar el usuario con más horas jugadas para ese género
    usuario_max_horas = horas_por_usuario[horas_por_usuario['genres'] == genero].nlargest(1, 'playtime_forever')
    
    if usuario_max_horas.empty:
            error_message = {"error": "No hay reseñas para el genero proporcionado."}
            return JSONResponse(status_code=404, content=error_message)
   
    # Paso 5: Calcular la acumulación de horas jugadas por año para ese usuario y género
    horas_por_anio = df_merged[df_merged['user_id'] == usuario_max_horas.iloc[0]['user_id']]
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
  

@app.get("/recomendacion_usuario/{id_usuario}")
def recomendacion_usuario(id_usuario : str):
    # Muestra aleatoria de datos (ajusta el tamaño según sea necesario)
    df = df_reviews.sample(n=5000, random_state=42)
    df_aux = df_reviews[df_reviews['user_id'] == id_usuario]
    if df_aux.empty:
            error_message = {"error": "No se encontro el usuario proporcionado."}
            return JSONResponse(status_code=404, content=error_message)
    df_aux = pd.DataFrame(df_aux.iloc[0]).T     
    df_combinada = pd.concat([df_aux, df], axis=0, ignore_index=True)
    # Eliminar duplicados en df_sample si es necesario
    df_combinada = df_combinada.drop_duplicates(subset=['user_id', 'item_id'])
    df_combinada.reset_index(drop=True,inplace=True)
     
    
    # Encontrar la fila de similitud del usuario deseado
    indice_usuario = df_combinada[df_combinada['user_id'] == id_usuario].index[0]
    # Crear una matriz de interacciones (usuarios x juegos)
    matriz_interacciones = pd.pivot_table(df_combinada, values='recommend', index='user_id', columns='item_id', fill_value=0)

    # Calcular la similitud del coseno entre usuarios
    similitud = cosine_similarity(matriz_interacciones)
    similitudes_usuario = similitud[indice_usuario]

    # Ordenar los usuarios por similitud en orden descendente y excluir al usuario de consulta
    usuarios_similares = sorted(range(len(similitudes_usuario)), key=lambda i: similitudes_usuario[i], reverse=True)[1:]

    # Obtener los juegos recomendados en función de los usuarios similares
    juegos_recomendados = []

    for usuario_similar in usuarios_similares:
        juegos_usuario_similar = matriz_interacciones.iloc[usuario_similar]
        juegos_recomendados_usuario_similar = juegos_usuario_similar[juegos_usuario_similar == 1].index.tolist()
        juegos_recomendados.extend(juegos_recomendados_usuario_similar)

    # Eliminar juegos que el usuario ya haya recomendado
    juegos_recomendados = list(set(juegos_recomendados) - set(matriz_interacciones.iloc[indice_usuario][matriz_interacciones.iloc[indice_usuario] == 1].index))

    # Limitar el número de recomendaciones
    juegos_recomendados = juegos_recomendados[:5]


    return juegos_recomendados
