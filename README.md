# Oyito
Neurona para identificar instrumentos 
Proyecto de oscilaciones y ondas, y herramientas computacionales 

En este proyecto se utilizan diversas librerías y se modifican códigos previamente implementados para formar un algoritmo que pueda aprender e identificar
instrumentos a partir de un entrenamiento de n épocas, dónde estudia el espectro de Fourier para cada muestra y luego clasifica alguna muestra que el usuario
quiera proporcionar.

# Requisitos
- Python
- Numpy
- torch
- torchaudio
- numpy
- matplotlib
- seaborn
- scikit-learn *(para `sklearn.metrics` y `sklearn.model_selection`)*
- tqdm

# Ejecucion del programa
* En el código encontrará diversas clases y funciones, dentro de las cuales están: Procesamiento de audio, información de los audios, modelo CNN, visualización, funciones de entrenamiento, cargar (archivos etiquetados) y predicción de audios nuevos. (En el caso particular se incluyó una extra para poder ver el espectro de Fourier de la muestra).

# ¿Cómo ejecutarlo?

(Antes de correr el código)
1. Los códigos tendrán ligeras variaciones pero en principio deberá modificar los siguientes parámetros para su funcionamiento:
    -Inicialmente la instalación de las bibliotecas previas.
    -Contar con una base de datos (ya sea la que se usó en este proyecto y estará reeferenciada en el github o una distinta). Importante que los udios esten en formato .wav o .mp3.

3. Dentro de las primeras lineas de código encontrará un apartado que dice "Configuración", es ahí donde tendrá que modificar lo siguiente
    EPOCHS: Esto determinará cuántas épocas quiere que entrene el algoritmo, si bien el más rápido en completar el entrenamiento fue el que utilizaba espectros de mel,  
     dentro de las pruebas no se superaron las 500 épocas.
    DATA-DIR: Aquí digitará la ruta en la que se encuentran sus archivos para el entrenamiento, asegurese de que sea una carpeta que siga esta estructura para evitar errores:
    Carpeta principal:
            - Instrumento 1: n muestras del instrumento
            - Instrumento 2: n muestras del instrumento
            - Instrumento 3: n muestras del instrumento
            ...

(Al correr el código)

3. Cuando corra el código se imprimirá un menú en la terminal el cuál tendrá las siguientes opciones
    1. Entrenar modelo
    * Cómo ya modificó los parámetros mencionados anteriormente, para el entrenamiento sólo debería seleccionar esta opción y esperar a que se complete.

    2. Clasificar audio
    * Esta es una opción post-entrenamiento, debido a que no funcionará si la red no tiene ya una memoria y datos para relacionar, sin embargo si quiere ahorrar el proceso de entrenamiento se le recomienda que descargue uno de los archivos "best_model_.pth" del github que son archivos de memoria de los entrenamientos realizados por el grupo de trabajo, para incluirlo dentro de la misma carpeta donde tenga el código de la red neuronal. Inportante no cambiar el nombre de este archivo. Fijarse bien a que entrenamiento corresponde el archivo descargado.

    --Si igualmente desea realizar el entrenamiento omita la explicación anterior, pues el algoritmo generará una memoria "best_model" al completar el entrenamiento.--

    Al seleccionar la opción de clasificación, se le pedirá que digite la ruta en la cuál está el aidio que desea clasificar, asegurese de digitarla completa hasta llegar al archivo .wav o .mp3. 
    *Importante* No es necesario que el audio de prueba pertenezca a la carpeta principal, este puede ser ajeno a la misma y tener una ubicación diferente

    3. Salir
    Para salir

    (4. Ver espectro de Fourier)
    Esta opción quedó implementada en el caso particular (+1 instrumento) y funciona de manera similar a la clasificación de audio, sólo debe escribir la ruta correspondiente del archivo, teniendo en cuenta las recomendaciones previas.

# Para correr el código caso particular
Este código es ligeramente diferente en estructura, sin embargo guarda las características principales mencionadas anteriormente.

*Consideraciones*

    - Este código funciona con 2 archivos de memoria, en el git encontrará archivos titulados "Caso particular", "best_model_1inst.pth" y "best_model_2inst.pth" los cuales corresponderán al entrenamiento realizado con las combinaciones de instrumentos y a los instrumentos individuales. Si desea correr este código deberá descargar ambos archivos de memoria.
    - Habrá algunas diferencias en la interfaz con el usuario, deberá digitar las épocas y la ruta directamente al ejecutar la opción 1 y se le pedirá decir si el entrenamiento se hará con tomas individuales o combinadas.
    - Para la clasificación, la funcionalidad es la misma.

Cómo última aclaración, el código del caso particular presenta algunas fallas que no pudieron solventarse y su funcionalidad no es la deseada, así que es el más propenso a presentar fallas si se llega a correr.

#Carpetas
*Audios*: en esta carpeta se encontrar los audios utilizados para probar el funcionamiento de las redes neuronales.
*Código*: en esta carpeta se encuentran los códigos documentados para cada caso.
*Memorias*: en esta carpeta se encuentran los mejores caminos, los cuales se obtuvieron al entrenar las redes neuronales.
*PDF*: en esta carpeta se encuentran parte de las referencias utilizadas.
*Graficas*: en esta carpeta se encuentran gráficas obtenidas para cada red neuronal con distintos cambios.

# Notas de Licencia
- Este proyecto se distribuye bajo la licencia MIT.
- Verifique las licencias originales antes de redistribuir código derivado.
