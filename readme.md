## Datasets

Por ahora trabajamos con dos datasets, el de chatbot arena y el de hardness.

El primero sale de [aca](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations/tree/main/data) y los datos son la comparacion entre dos respuestas de modelos de IA a un mismo prompt. Por ahora lo usamos para calcular la diferencia de tiempo entre las consultas asi podemos determinar la FDP del Intervalo de Arribo.

El [segundo]() surge a partir del primero pero fue transformado previamente para calcular el "hardness" (dificultad para responder a ese prompt) a partir de la media de scores que haya para ese prompt. Ej: Si un modelo le dio un score de 9 y otro de 7 el hardness para ese prompt es un 8 por ser el promedio entre los dos. Ademas a partir de este dataset se calcula la cantidad de tokens del prompt (tokens de entrada) y el promedio de tokens de salida (saca la media entre los tokens de salida de cada respuesta).

## Modelo

Por ahora nuestro modelo funciona de la siguiente manera, es un simil a un sistema de colas como los que vimos en clase.

El intervalo de arribos va a indicar la llegada de nuestro proximo prompt, la cantidad de tokens y el hardness tambien van a ser en base a la FDP que extraigamos del analisis del segundo dataset. Segun el hardness, vamos a determinar a que modelo le vamos a enrutar el prompt que llego (los umbrales $h_1$ y $h_2$ son nuestas variables de control), cuando haya que procesar ese prompt, por medio de otra FDP, vamos a calcular la cantidad de tokens de salida y vamos a calcular el proximo tiempo de salida para la cola de ese modelo.
Contamos con 3 modelos distintos (uno por cada nivel de dificultad: bajo, medio, alto) según el modelo al que le toque el prompt cambian los parámetros en la subrutina de "tiempo de atención".

## Scripts

Antes de ejecutar alguno de los scripts, hay que tener instalado python (para probar localmente, esto despues lo movemos a un google colab) e instalar las siguientes librerias por medio de este comando:

```python
pip install pandas numpy scipy matplotlib tiktoken datasets fitter
```

El primer script que debemos ejecutar es `preprocessing.p` para que nos genere los archivos `interarrivals.csv` y `consultas.csv`. A partir de estos dos se realizan las aproximaciones con el script `distributions.py` de distribuciones de probabilidad para aplicar cualquiera de los dos metodos vistos en clase para variables aleatorias.
Para ejecutar cualquiera de ellos basta con escribir (dentro de la carpeta source):

```python
py preprocessing.py

# o

py distributions.py

```

## Pasos

1. Primero tenemos que hacer el analisis de los datos para obtener las FDP correspondientes. Las FDP necesarias son: Intervalo de arribos, tokens de entrada, hardness y tokens de salida. Siempre las FDP que se utilicen tienen que poder resolverse por alguno de los dos metodos que vimos en clase (funcion inversa o metodo del rechazo), si la FDP que mejor se ajusta no se puede resover por alguno de esos metodos pasamos a la siguiente aproximacion.
2. Definir bien nuestras variables de estado y de resultado para saber como manipularlas dentro de las subutinas
3. Una vez completado lo anterior, usando codigo de python (un ciclo while hasta que se alcance el tiempo final de simuacion y llamados a subrutinas) hay que replicar la simulacion evento a evento como vimos en clase incluyendo vaciamiento (lo podemos discutir)
4. Por ultimo habria que redactar el paper
5. Mover el contenido del repo a un google colab (Era más facil hacerlo ahi desde un inicio pero no permitia que todos editemos codigo a la vez asi que se iba a volver engorroso, usamos vscode con la extension live share)
6. Armar la presentacion

Los ultimos 3 pasos se pueden hacer en paralelo
