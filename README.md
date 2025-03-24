# Customer Satisfaction Analysis - Sentiment Analysis using Deep Learning and Python

Se presenta a continuación un Análisis de Sentimiento para frases de texto mediante la librería Transformers de HuggingFace,
que permite trabajar con Procesamiento de Lenguaje Natural.
HuggingFace ofrece en su sitio de Internet modelos de Deep Learning entrenados para NLP, además de datasets en varios idiomas.
Además ha desarrollado un API que permite utilizar AI con unas pocas líneas de código.
En este ejercicio se analiza los textos correspondientes a tres opiniones: una claramente positiva, otra claramente negativa, y la tercera de carácter ambivalente.
Se utiliza cuatro modelos distintos preentrenados para NLP y se compara su nivel de pecisión en el análisis del sentimiento.

## CÓDIGO PYTHON:

### PRIMER ANÁLISIS: utilizando el modelo "BERT BASE MULTILINGUAL UNCASED SENTIMENT":
```
from transformers import pipeline

clasificador = pipeline('sentiment-analysis',    
                      model="nlptown/bert-base-multilingual-uncased-sentiment")

frases = ["¡Me encantan los artículos de Medium, son lo máximo!", 
          "Odio los lunes, no sirven para nada.", 
          "El libro es medio bueno; me gustaron algunos personajes."]

resultados = clasificador(frases)

for result in resultados:
    print(f"polaridad: {result['label']}, score: {round(result['score'], 2)}")
```
OUTPUT:

polaridad: 5 stars, score: 0.91

polaridad: 1 star,  score: 0.91

polaridad: 3 stars, score: 0.77

### SEGUNDO ANÁLISIS: utilizando el modelo "ROBERTUITO":
```
from transformers import pipeline

clasificador = pipeline('sentiment-analysis',    
                      model="pysentimiento/robertuito-sentiment-analysis")

frases = ["¡Me encantan los artículos de Medium, son lo máximo!", 
          "Odio los lunes, no sirven para nada.", 
          "El libro es medio bueno; me gustaron algunos personajes."]

resultados = clasificador(frases)

for result in resultados:
    print(f"polaridad: {result['label']}, score: {round(result['score'], 2)}")
```
OUTPUT:

polaridad: POS, score: 0.98

polaridad: NEG, score: 0.97

polaridad: POS, score: 0.96

### TERCER ANÁLISIS: utilizando el modelo "BERT BASE SPANISH WWM CASED":
```
from transformers import pipeline

clasificador = pipeline('sentiment-analysis',    
                      model="dccuchile/bert-base-spanish-wwm-cased")

frases = ["¡Me encantan los artículos de Medium, son lo máximo!", 
          "Odio los lunes, no sirven para nada.", 
          "El libro es medio bueno; me gustaron algunos personajes."]

resultados = clasificador(frases)

for result in resultados:
    print(f"polaridad: {result['label']}, score: {round(result['score'], 2)}")
```
OUTPUT:

polaridad: LABEL_1, score: 0.53

polaridad: LABEL_0, score: 0.52

polaridad: LABEL_1, score: 0.5

### CUARTO ANÁLISIS: utilizando el modelo "BERT BASE SPANISH WWM UNCASED":
```
from transformers import pipeline

clasificador = pipeline('sentiment-analysis',    
                      model="dccuchile/bert-base-spanish-wwm-uncased")

frases = ["¡Me encantan los artículos de Medium, son lo máximo!", 
          "Odio los lunes, no sirven para nada.", 
          "El libro es medio bueno; me gustaron algunos personajes."]

resultados = clasificador(frases)

for result in resultados:
    print(f"polaridad: {result['label']}, score: {round(result['score'], 2)}")
```
OUTPUT:

polaridad: LABEL_1, score: 0.55

polaridad: LABEL_1, score: 0.50

polaridad: LABEL_0, score: 0.54

### CONCLUSIÓN

La comparación de los resultados de la utilización de los cuatro modelos preentrenados para NLP para el análisis de sentimiento aplicado a las tres opiniones de ejemplo permite concluir que el modelo preentrenado más preciso es el primero, el modelo "BERT BASE MULTILINGUAL UNCASED SENTIMENT", ya que identifica claramente a la primera frase como muy positiva (5/5 stars), a la segunda como muy negativa (1/5 stars), y a la tercera como ambivalente (3/5 stars).
