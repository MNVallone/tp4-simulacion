## Datasets

Por ahora trabajamos con dos datasets, el de chatbot arena y el de hardness.

El primero sale de [aca](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations/tree/main/data) y los datos son la comparacion entre dos respuestas de modelos de IA a un mismo prompt. Por ahora lo usamos para calcular la diferencia de tiempo entre las consultas asi podemos determinar la FDP del Intervalo de Arribo.

El [segundo]() surge a partir del primero pero fue transformado previamente para calcular el "hardness" (dificultad para responder a ese prompt) a partir de la media de scores que haya para ese prompt. Ej: Si un modelo le dio un score de 9 y otro de 7 el hardness para ese prompt es un 8 por ser el promedio entre los dos. Ademas a partir de este dataset se calcula la cantidad de tokens del prompt (tokens de entrada) y el promedio de tokens de salida (saca la media entre los tokens de salida de cada respuesta).

## Modelo

- Generación del arribo: Se determina el momento exacto en el que ingresa un nuevo prompt al sistema calculando el avance del reloj mediante la FDP del intervalo de arribos.
- Asignación de atributos: Inmediatamente al llegar se usan las FDP para definir las características particulares de ese prompt: su nivel de dificultad (hardness) y sus tokens asociados (cada una con su FDP).
- Enrutamiento (Routing): Se compara el hardness del prompt contra los umbrales de decisión o variables de control ($h_1$ y $h_2$). Según dónde caiga ese valor, el prompt es derivado a uno de los 3 modelos de IA (servidores de dificultad baja, media o alta).
- Ingreso a la cola: El prompt se encola en el servidor asignado. Si el modelo de IA está libre, pasa directamente a ser procesado; si está ocupado, espera su turno.
- Procesamiento (Tiempo de servicio): Al momento de la atención, se calcula cuánto tardará el modelo en responder. Este tiempo de servicio se determina combinando la cantidad de tokens de salida (mediante su respectiva FDP) y los parámetros de rendimiento específicos del modelo asignado.
- Programación de la salida: Con el tiempo de servicio calculado, se proyecta en el reloj de simulación el tiempo de salida de ese modelo, indicando el momento exacto en el que el prompt termina de ser procesado y abandona el sistema.

### TEI

| Evento | EFNC | EFC | Condicion |
|-----------|-----------|-----------|-----------|
| $Llegada$    | $Llegada$    | $Salida_i$    | $NP_{global} = 1$|
| $Salida_i$    | -    | $Salida_i$   |$NP_i > 0$|

## Scripts

Antes de ejecutar alguno de los scripts, hay que tener instalado python (para probar localmente, esto despues lo movemos a un google colab) e instalar las siguientes librerias por medio de este comando:

```python
pip install pandas numpy scipy matplotlib tiktoken datasets fitter
```

El primer script que debemos ejecutar es `preprocessing.p` para que nos genere los archivos `interarrivals.csv` y `consultas.csv`. A partir de estos dos se realizan las aproximaciones con el script `distributions.py` de distribuciones de probabilidad para aplicar cualquiera de los dos metodos vistos en clase para variables aleatorias.
Para ejecutar cualquiera de ellos basta con escribir (dentro de la carpeta source):

```python
py preprocessing.py
# y luego
py distributions.py
```

## Pasos

1. Primero tenemos que hacer el analisis de los datos para obtener las FDP correspondientes. Las FDP necesarias son: Intervalo de arribos, tokens de entrada, hardness y tokens de salida. Siempre las FDP que se utilicen tienen que poder resolverse por alguno de los dos metodos que vimos en clase (funcion inversa o metodo del rechazo), si la FDP que mejor se ajusta no se puede resover por alguno de esos metodos pasamos a la siguiente aproximacion.
2. Definir bien nuestras variables de estado y de resultado para saber como manipularlas dentro de las subutinas
3. Una vez completado lo anterior, usando codigo de python (un ciclo while hasta que se alcance el tiempo final de simuacion y llamados a subrutinas) hay que replicar la simulacion evento a evento como vimos en clase incluyendo vaciamiento (lo podemos discutir)
4. Por ultimo habria que redactar el paper
5. Mover el contenido del repo a un google colab (Era más facil hacerlo ahi desde un inicio pero no permitia que todos editemos codigo a la vez asi que se iba a volver engorroso, usamos vscode con la extension live share)
6. Armar la presentacion

_Nota: Los últimos 3 pasos se pueden hacer en paralelo_

## Modelos de IA para la simulación

Los modelos de IA que vamos a estar simulando son GPT-5.6 Luna (sería el bajo), GPT-5.6 Terra (medio) y GPT-5.6 Sol (alto). Cada uno cuenta con datos de velocidad y precio que obtuvimos de distintas fuentes. Para representar tres niveles de capacidad se utilizan modelos de la familia GPT-5.6 de OpenAI en modo non-reasoning.

<table> <thead> <tr> <th>Recurso</th> <th>Modelo</th> <th>TTFT</th> <th>Velocidad</th> <th>Costo input / 1M tokens</th> <th>Costo output / 1M tokens</th> </tr> </thead> <tbody> <tr> <td>Small</td> <td>GPT-5.6 Luna</td> <td>0.93 s</td> <td>98.1 tokens/s</td> <td>USD 0.20</td> <td>USD 1.20</td> </tr> <tr> <td>Medium</td> <td>GPT-5.6 Terra</td> <td>1.00 s</td> <td>73.8 tokens/s</td> <td>USD 2.00</td> <td>USD 12.00</td> </tr> <tr> <td>Large</td> <td>GPT-5.6 Sol</td> <td>1.17 s</td> <td>60 tokens/s</td> <td>USD 4.00</td> <td>USD 20.00</td> </tr> </tbody> </table>

Los valores de TTFT y velocidad de salida se toman como parametros de referencia obtenidos a partir de benchmarks de la API. Los costos corresponden al precio por tokens de cada modelo.

### Precios y especificaciones de los modelos

Los precios por token y las caracteristicas generales de los modelos se obtienen de la documentacion oficial de OpenAI.

OpenAI - API Pricing<br>
Precios oficiales por 1 millon de tokens de entrada y salida para GPT-5.6 Luna, Terra y Sol.<br>
https://developers.openai.com/api/docs/pricing<br>
OpenAI - GPT-5.6 Luna<br>
https://developers.openai.com/api/docs/models/gpt-5.6-luna<br>
OpenAI - GPT-5.6 Terra<br>
https://developers.openai.com/api/docs/models/gpt-5.6-terra<br>
OpenAI - GPT-5.6 Sol<br>
https://developers.openai.com/api/docs/models/gpt-5.6-sol<br>

### Velocidad y latencia<br>

Los valores de TTFT (Time To First Token) y velocidad de salida en tokens por segundo se obtienen de Artificial Analysis, que realiza benchmarks independientes sobre las APIs de los proveedores.<br>

Artificial Analysis - GPT-5.6 Luna<br>
https://artificialanalysis.ai/models/releases/gpt-5-6-luna<br>
Artificial Analysis - GPT-5.6 Terra<br>
https://artificialanalysis.ai/models/releases/gpt-5-6-terra<br>
Artificial Analysis - GPT-5.6 Sol<br>
https://artificialanalysis.ai/models/releases/gpt-5-6-sol<br>
Artificial Analysis - Metodologia de benchmarking<br>
https://artificialanalysis.ai/methodology/performance-benchmarking<br>

### Tiempo de atencion

El tiempo de atencion de una consulta se calcula mediante:

$$
T_m = TTFT_m + \frac{OutputTokens}{Velocidad_m}
$$

Donde:

T_m: tiempo de atencion del modelo seleccionado, en segundos.
TTFT_m: tiempo hasta recibir el primer token.
OutputTokens: cantidad de tokens generados en la respuesta.
Velocidad_m: velocidad de generacion del modelo, en tokens por segundo.  

Costo de una consulta

El costo depende tanto de los tokens de entrada como de los tokens de salida:

$$
Costo_m =
\frac{InputTokens}{1.000.000} \times PrecioInput_m
+
\frac{OutputTokens}{1.000.000} \times PrecioOutput_m
$$

Donde:

InputTokens: cantidad de tokens del prompt.
OutputTokens: cantidad de tokens de la respuesta.
PrecioInput_m: precio por un millon de tokens de entrada del modelo.
PrecioOutput_m: precio por un millon de tokens de salida del modelo.

De esta forma, cada consulta tiene un tiempo de atencion y un costo diferente dependiendo del modelo al que sea asignada.
