![Pandas](https://img.shields.io/badge/-Pandas-333333?style=flat&logo=pandas)
![Numpy](https://img.shields.io/badge/-Numpy-333333?style=flat&logo=numpy)
![Matplotlib](https://img.shields.io/badge/-Matplotlib-333333?style=flat&logo=matplotlib)
![Seaborn](https://img.shields.io/badge/-Seaborn-333333?style=flat&logo=seaborn)
![Scikitlearn](https://img.shields.io/badge/-Scikitlearn-333333?style=flat&logo=scikitlearn)
![FastAPI](https://img.shields.io/badge/-FastAPI-333333?style=flat&logo=fastapi)
![Render](https://img.shields.io/badge/-Render-333333?style=flat&logo=render)
# API_MLOps
![](/img/MLOps.png)
## Introduccion

En este proyecto combinamos el perfil de un Data Engineer y Data Scientist, para la plataforma  de juegos Steam. Para su desarrollo, se entregan unos datosSet y se solicita una API deployada en un servicio en la nube y la aplicación de dos modelos de Machine Learning, por una lado, un análisis de sentimientos sobre los comentarios de los usuarios de los juegos y, por otro lado, la recomendación de juegos a partir de los gustos de un usuario en particular.

## Desarrollo de la Api
Se ecuentra en el siguiente link: [Api  Link](https://api-mlops.onrender.com)
<br><br>

Para el desarrolo de la API se decidió utilizar FastAPI, creando las siguientes funciones:


def PlayTimeGenre( genero : str ): Debe devolver año con mas horas jugadas para dicho género.
Ejemplo de retorno: {"Año de lanzamiento con más horas jugadas para Género X" : 2013}

def UserForGenre( genero : str ): Debe devolver el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año.
Ejemplo de retorno: {"Usuario con más horas jugadas para Género X" : us213ndjss09sdf, "Horas jugadas":[{Año: 2013, Horas: 203}, {Año: 2012, Horas: 100}, {Año: 2011, Horas: 23}]}

def UsersRecommend( año : int ): Devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado. (reviews.recommend = True y comentarios positivos/neutrales)
Ejemplo de retorno: [{"Puesto 1" : X}, {"Puesto 2" : Y},{"Puesto 3" : Z}]

def UsersNotRecommend( año : int ): Devuelve el top 3 de juegos MENOS recomendados por usuarios para el año dado. (reviews.recommend = False y comentarios negativos)
Ejemplo de retorno: [{"Puesto 1" : X}, {"Puesto 2" : Y},{"Puesto 3" : Z}]

def sentiment_analysis( año : int ): Según el año de lanzamiento, se devuelve una lista con la cantidad de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento.
Ejemplo de retorno: {Negative = 182, Neutral = 120, Positive = 278}

def recomendacion_usuario( id de usuario ): Ingresando el id de un usuario, deberíamos recibir una lista con 5 juegos recomendados para dicho usuario.

## Transformación de Datos
Para este proyecto se proporcionaron tres archivos JSON:

<b>australian_user_reviews.json:</b> dataset que contiene los comentarios que los usuarios realizaron sobre los juegos, además de el id del usuario, url del perfil y el id del juego que comenta.

<b>australian_users_items.json:</b> dataset que contiene información sobre los juegos, así como el tiempo acumulado que cada usuario jugó a un determinado juego.

<b>output_steam_games.json:</b>  dataset que contiene datos relacionados con los juegos en sí, como los título, el desarrollador, los precios, características técnicas, etiquetas, entre otros datos.

<br><br>
En esta etapa, se crean 3 archivos [ETL_games](etl_games.ipynb),  [ETL_items](etl_items.ipynb),  [ETL_reviews](etl_Reviews.ipynb), para leer y transformar cada uno de los dataset entregados. Se lee los [.Json](https://drive.google.com/drive/folders/1b1xrbXw88Ua4KWiAl47YyuTSzj3XHa_R?usp=sharing) y se los convierte a .parquet para que ocupe menos almacenamiento. Se desanidan los dataset, se eliminan las columnas irrelevantes y tambien filas que en mayoria contien NaN. Se realiza algunas convercines de tipos para operar y para almacenar, y se eliminan duplicados para optimizar el rendimiento de la API, teneniendo en cuenta las limitaciones de almacenamiento del deploy. Para las transformaciones se utilizó la librería Pandas. :panda_face:
<br><br>
## Feature engineering
Se pide aplicar un análisis de sentimiento a los reviews de los usuarios. Para ello se creó una nueva columna llamada 'sentiment_analysis' que reemplaza a la columna que contiene los reviews donde clasifica los sentimientos de los comentarios con la siguiente escala:
<br> <br>
* 0 si es malo
* 1 si es neutral o esta sin review
* 2 si es positivo 
<br> <br>
Se realiza un análisis de sentimiento básico utilizando TextBlob que es una biblioteca de procesamiento de lenguaje natural (NLP) en Python. El objetivo de esta metodología es asignar un valor numérico a un texto, en este caso a los reviews que los usuarios dejaron para un juego, y asi representar si el sentimiento expresado en el texto es negativo, neutral o positivo.

 En este caso, se consideraron las polaridades por defecto del modelo, el cuál utiliza umbrales -0.2 y 0.2, siendo polaridades negativas por debajo de -0.2, positivas por encima de 0.2 y neutrales entre medio de ambos.

 <br><br>

 ## Análisis exploratorio de los datos
 Una vez que concluimos el proceso de ETL empezamos con esta etapa. Donde verificamos algunos campos y mas que nada realizamos graficas y algunos promedios para ver la agrupacion de los datos. A veces con subconjuntos de los datos dado su costo computacional. Donde tomaos la decision de usar un subconjunto de datos para la Api dado los escasos recursos con los que contamos. Raaaecortamos varias veces los dataset a usar hasta llegar a un minimo que nos permitio deployar el proyecto. Por lo tal no se encontraran algunos juegos, usuarios y las cantidades de horas y otras medidas se veran influenciadas. 
  
 ## Modelo de aprendizaje automático
 Aquí el input es un juego y el output es una lista de juegos recomendados, para ello se aplica la similitud del coseno. La  recomendación debe aplicar el filtro user-item, esto es tomar un usuario, se encuentran usuarios similares y se recomiendan ítems que a esos usuarios similares les gustaron. En este caso el input es un usuario y el output es una lista de juegos que se le recomienda a ese usuario, en general se explican como “A usuarios que son similares a tí también les gustó…”

Si es un sistema de recomendación user-item:

def recomendacion_usuario( id de usuario ): Ingresando el id de un usuario, deberíamos recibir una lista con 5 juegos recomendados para dicho usuario.

## Deploy
Para el deploy de la API se seleccionó la plataforma Render que es una nube unificada para crear y ejecutar aplicaciones y sitios web, permitiendo el despliegue automático desde GitHub, y dado que usamos el servicio gratuito y este cuenta con una limitada capacidad, se debe ajustar los archivos y las operaciones para lograr su funcionamiento. Comparto nuevamente el link a la api: https://api-mlops.onrender.com
 <br><br>
 
 ## Video
https://youtu.be/QIF2AuOpboY
 ## Dataset
 Estos  [Archivos](https://drive.google.com/drive/folders/1b1xrbXw88Ua4KWiAl47YyuTSzj3XHa_R?usp=sharing) deben descargarse, descomprimirse y guardarlos dentro de la carpeta data del proyecto. Fueron excluidos de este repositorio para no sobrecargar al sitio donde alojamos la api.



### Maciel Norman
![](/data/github.ico)  https://github.com/Norman2022


