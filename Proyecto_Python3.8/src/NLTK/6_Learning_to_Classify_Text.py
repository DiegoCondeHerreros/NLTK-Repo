# -*- coding: utf-8 -*-
'''
Created on 8 mar. 2021

@author: DIEGO
'''
#La deteccion de patrones es importante porque la estructura de las palabras y la frecuencia de las mismas se correlaciona con aspectos de su significado
#¿Como podemos identificar las caracteristicas del lenguaje que se pueden clasificar?
#¿Como podemos construir modelos de lenguaje que realicen tareas de procesamiento automaticamente?
#¿Que podemos aprender del lenguaje de estos modelos?

#1. Clasificación Supervisada
#La clasificación es la tarea de elegir la etiqueta de clase correcta para una entrada, cada input se considera de forma aislada del resto de inputs
#La clasificación es supervisada si se construye basandose en corpora de entrenamiento que tienen la asignacion correcta de etiquetas 

#1.1 Identificación de género
#Nombres que acaban en a,e,i suelen ser femeninos y los que acaban en k,o,r,s,t suelen ser masculinos
#Se deben de extraer las características relevantes y se codifican
#Creamos un extractor de características que construye un diccionario con información relevante
#def gender_features(word):
#    return {'last_letter':word[-1]}
#print(gender_features('Shrek'))
#A lo que devuelve lo llamamos un conjunto de caracteristicas
#Preparamos ista de ejemplos y etiquetas de clase
import nltk
from nltk.corpus import names
#labeled_names=([(name,'male') for name in names.words('male.txt')] +
#               [(name,'female') for name in names.words('female.txt')])
import random
#random.shuffle(labeled_names)
#A continuacion dividimos los datos en un conjunto de entrenamiento y un conjunto de prueba, los usamos para entrenar un clasificador Naive Bayes
#featuresets=[(gender_features(n),gender) for (n,gender) in labeled_names]
#train_set,test_set = featuresets[500:],featuresets[:500]
#classifier=nltk.NaiveBayesClassifier.train(train_set)
#Le damos uso
#print(classifier.classify(gender_features('Neo')))
#print(classifier.classify(gender_features('Trinity')))
#print(nltk.classify.accuracy(classifier,test_set))
#Con esto vemos las palabras que contienen información más relevante.
#classifier.show_most_informative_features(5)
#Ahora vamos a intentar lo mismo pero con la longitud del nombre
#def gender_features0(word):
#    return {'name_length':len(word)}
#print(gender_features0('Shrek'))
#featuresets=[(gender_features0(n),gender) for (n,gender) in labeled_names]
#train_set,test_set = featuresets[500:],featuresets[:500]
#classifier=nltk.NaiveBayesClassifier.train(train_set)
#print(classifier.classify(gender_features0('Neo')))
#print(classifier.classify(gender_features0('Trinity')))
#print(nltk.classify.accuracy(classifier,test_set))
#classifier.show_most_informative_features(5)
#Ahora vamos a intentarlo con la primera letra en vez de la última
#def gender_features1(word):
#    return {'first_letter':word[0]}
#print(gender_features1('Shrek'))
#featuresets=[(gender_features1(n),gender) for (n,gender) in labeled_names]
#train_set,test_set = featuresets[500:],featuresets[:500]
#classifier=nltk.NaiveBayesClassifier.train(train_set)
#print(classifier.classify(gender_features1('Neo')))
#print(classifier.classify(gender_features1('Trinity')))
#print(nltk.classify.accuracy(classifier,test_set))
#classifier.show_most_informative_features(5)
#Para ahorrar memoria se puede hacer esto
from nltk.classify import apply_features
#train_set=apply_features(gender_features,labeled_names[500:])
#test_set=apply_features(gender_features,labeled_names[:500] )

#1.2 Escogiendo las características correctas
#De cara a elegir se suele tirar de intuición y se prueban todas las características que se te puedan ocurrir y ver que funciona mejor
#En este prograna miramos si están ciertas letras y su frecuencia de aparición
#def gender_features2(name):
#    features={}
#    features["first_letter"]=name[0].lower()
#    features["last_letter"]=name[-1].lower()
#    for letter in 'abcdefghijklmnopqrstuvwxyz':
#        features["count({})".format(letter)]= name.lower().count(letter)
#        features["has({})".format(letter)]=(letter in name.lower())
#    return features
#print(gender_features2('John'))
#Overfitting llamamos a cuando hay un exceso de características que impide el correcto funcionamiento del clasificador
#En este ejemplo anterior el exceso de características lleva a un clasificador que tiene un 1% menos de eficiencia que uno que solo usa la última letra.
#featuresets=[(gender_features2(n),gender) for (n,gender) in labeled_names]
#train_set,test_set=featuresets[500:],featuresets[:500]
#classifier=nltk.NaiveBayesClassifier.train(train_set)
#print(nltk.classify.accuracy(classifier,test_set))
#Para refinar el conjunto de características se usa el analisis de errores. Primero se elige un conjunto de desarrollo que contiene el corpus, este se divide en el conjunto de entrenamiento y el dev-test set
#Divimos el total de nombres en estos tres conjuntos
#train_names=labeled_names[1500:]
#devtest_names=labeled_names[500:1500]
#test_names=labeled_names[:500]
#Conjunto de entrenamiento se usa para entrenar el modelo
#Dev-test se usa para hacer análisis de errores
#test-set sirve para la evaluación final del modelo   
#El conjunto de entrenamiento + el de dev-test se consideran el conjunto de desarrollo
#Le pasamos el extractor de características a los tres conjuntos
#train_set=[(gender_features(n),gender) for (n,gender) in train_names]
#devtest_set=[(gender_features(n),gender) for (n,gender) in devtest_names]
#test_set=[(gender_features(n),gender)for(n,gender)in test_names]
#Clasificamos el conjunto de entrenamiento
#classifier = nltk.NaiveBayesClassifier.train(train_set)
#Con este conjunto anterior y el de dev_test obtenemos el rendimiento
#print(nltk.classify.accuracy(classifier,devtest_set)) 
#Obtenemos la lista de errores
#errors=[]
#for(name,tag)in devtest_names:
#    guess=classifier.classify(gender_features(name))
#    if guess!=tag:
#        errors.append((tag,guess,name)) 
#Examinamos los errores para ver que se puede mejorar        
#for(tag,guess,name)in sorted(errors):
#    print('correct={:<8} guess={:<8s} name={:<30}'.format(tag,guess,name))
#De esto obtenemos que los nombres que acaban en yn suelen ser femeninos a pesar de que los que acaben en n son masculinos y ch masculinso cuando h son femeninos
#def gender_features3(word):
#    return {'suffix1':word[-1:],
#            'suffix2':word[-2:]}
#train_set = [(gender_features(n), gender) for (n, gender) in train_names]
#devtest_set = [(gender_features(n), gender) for (n, gender) in devtest_names]
#classifier = nltk.NaiveBayesClassifier.train(train_set)
#print(nltk.classify.accuracy(classifier, devtest_set))
#Este lo hace mejor y esto es un proceso iterativo hasta tener una puntuación casi perfecta
#Se debe de ir probando diferentes conjuntos de datos para entrenamiento y test

#1.3 Clasificación de Documentos 
#Con los documentos de las películas los categorizamos con reseñas positivas y negativas
#from nltk.corpus import movie_reviews
#documents=[(list(movie_reviews.words(fileid)),category)
#           for category in movie_reviews.categories()
#           for fileid in movie_reviews.fileids(category)]
#random.shuffle(documents)
#Creamos extractor de características y para limitar accedemos unicamente a las 2000 palabras más frecuentes
#El extractor comprueba si las palabras están presentes en el documento
#all_words=nltk.FreqDist(w.lower() for w in movie_reviews.words())
#word_features=list(all_words)[:2000]
#def document_features(document):
#    document_words=set(document)
#    features={}
#    for word in word_features:
#            features['contains({})'.format(word)] = (word in document_words)
#    return features
#print(document_features(movie_reviews.words('pos/cv957_8737.txt')))
#El motivo por el que utilizamos un conjunto en vez de una lista es porque es más rápida la búsqueda
#Ahora entrenamos clasificador 
#featuresets=[(document_features(d),c)for(d,c) in documents]
#train_set,test_set=featuresets[100:],featuresets[:100]
#classifier=nltk.NaiveBayesClassifier.train(train_set)
#print(nltk.classify.accuracy(classifier,test_set))
#classifier.show_most_informative_features(5) 
 
#1.4 Part-of-Speech Tagging 
#Hacemos clasificador que encuentra que sufijos son los más comunes
#from nltk.corpus import brown
#suffix_fdist=nltk.FreqDist()
#for word in brown.words():
#    word=word.lower()
#    suffix_fdist[word[-1:]] +=1
#    suffix_fdist[word[-2:]] +=1
#    suffix_fdist[word[-3:]] +=1
#common_suffixes=[suffix for (suffix,count)in suffix_fdist.most_common(100)]
#print(common_suffixes)
#Extractores de características que busca estos sufijos en una palabra
#def pos_features(word):
#    features={}
#    for suffix in common_suffixes:
#        features['endswith({})'.format(suffix)]=word.lower().endswith(suffix)
#    return features
#Ahora entrenamos el clasificador
#tagged_words=brown.tagged_words(categories='news')
#featuresets=[(pos_features(n),g)for(n,g)in tagged_words]
#size=int(len(featuresets)*0.1)
#train_set,test_set=featuresets[size:],featuresets[:size]
#classifier=nltk.DecisionTreeClassifier.train(train_set)
#print(nltk.classify.accuracy(classifier,test_set))
#print(classifier.classify(pos_features('cats')))
#Se puede imprimir el arbol de decision como pseudocódigo
#print(classifier.pseudocode(depth=4))
   
#1.5 Explotando el contexto 
#Para emplear el contexto modificamos el extractor de características de tal modo que pille otras
#def pos_features(sentence,i):
#    features={"suffix(1)":sentence[i][-1:],
#              "suffix(2)":sentence[i][-2:],
#              "suffix(3)":sentence[i][-3:]}
#    if i == 0:
#        features["prev-word"]="<START>"
#    else:
#        features["prev-word"]=sentence[i-1]
#    return features

from nltk.corpus import brown
from unicodedata import category
#print(pos_features( brown.sents()[0],8))
#tagged_sents = brown.tagged_sents(categories='news')    
#featuresets=[]
#for tagged_sent in tagged_sents:
#    untagged_sent = nltk.tag.untag(tagged_sent)
#    for i, (word,tag) in enumerate(tagged_sent):
#        featuresets.append((pos_features(untagged_sent,i),tag))
#size=int(len(featuresets)*0.1)
#train_set,test_set = featuresets[size:],featuresets[:size]
#classifier=nltk.NaiveBayesClassifier.train(train_set)
#print(nltk.classify.accuracy(classifier,test_set))

#1.6 Clasificación de secuencias
#Consiste en que clasifica el input y esto le sirve para encontrar la etiqueta de la siguiente palabra
#Esto se hace añadiendo variable history que guarda las tags     
#def pos_features(sentence,i,history):
#    features = {"suffix(1)":sentence[i][-1:],
#                "suffix(2)":sentence[i][-2:],
#                "suffix(3)":sentence[i][-3:]}
#    if i==0:
#        features["prev-word"]="<START>"
#        features["prev-tag"]="<START>"
#    else:
#        features["prev-word"]=sentence[i-1]
#        features["prev-tag"]=history[i-1]
#    return features

#class ConsecutivePosTagger(nltk.TaggerI):

#   def __init__(self, train_sents):
#        train_set = []
#        for tagged_sent in train_sents:
#            untagged_sent = nltk.tag.untag(tagged_sent)
#            history = []
#            for i, (word, tag) in enumerate(tagged_sent):
#                featureset = pos_features(untagged_sent, i, history)
#                train_set.append( (featureset, tag) )
#                history.append(tag)
#        self.classifier = nltk.NaiveBayesClassifier.train(train_set)

#    def tag(self, sentence):
#        history = []
#        for i, word in enumerate(sentence):
#            featureset = pos_features(sentence, i, history)
#            tag = self.classifier.classify(featureset)
#            history.append(tag)
#        return zip(sentence, history)
    
#tagged_sents = brown.tagged_sents(categories='news')   
#size=int(len(tagged_sents)*0.1)
#train_sents,test_sents = tagged_sents[size:],tagged_sents[:size]
#tagger = ConsecutivePosTagger(train_sents)
#print(tagger.evaluate(test_sents))


#2 Más Ejemplos de Clasificación Supervisada

#2.1 Segmentacion de oraciones
"""
sents=nltk.corpus.treebank_raw.sents()
tokens=[]
boundaries=set()
offset=0
for sent in sents:
    tokens.extend(sent)
    offset += len(sent)
    boundaries.add(offset-1)
#Aquí hemos guardado la frase como lista de tokens y el boundaries como lista de indices donde está la puntuación
def punct_features(tokens,i):
    return{'next-word-capitalized':tokens[i+1][0].isupper(),
           'prev-word':tokens[i-1].lower(),
           'punct':tokens[i],
           'prev-word-is-one-char':len(tokens[i-1])==1}
#Extraemos los diferentes signos de puntuación y ponemos si marca el fin de una oración o no.
featuresets=[(punct_features(tokens,i),(i in boundaries))
             for i in range(1,len(tokens)-1)
             if tokens[i] in '.?!']
#Entrenamos y evaluamos clasificador
size = int(len(featuresets) * 0.1)    
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))
#Otro segmentador de oraciones 
def segment_sentences(words):
    start = 0
    sents = []
    for i, word in enumerate(words):
        if word in '.?!' and classifier.classify(punct_features(words, i)) == True:
            sents.append(words[start:i+1])
            start = i+1
    if start < len(words):
        sents.append(words[start:])
    return sents
"""
#2.2 Identifying Dialogue Act Types
#Una manera de entender una conversación es entender los "dialogue acts" que permiten entender de verdad la conversación
#En los NPS Chat Corpus tiene mensajes de chats y están etiquetados con 15 etiquetas de diálogo
#estos están en xml y usa metodo correspondiente para tratarlo
"""
posts=nltk.corpus.nps_chat.xml_posts()[:10000]
#Definimos el extractor de características que comprueba las palabras que contiene 
def dialogue_act_features(post):
    features={}
    for word in nltk.word_tokenize(post):
        features['contains({})'.format(word.lower())]=True
    return features
featuresets=[(dialogue_act_features(post.text),post.get('class'))
             for post in posts]
size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))

#2.3 Reconociendo vinculación textual
#Consiste e determinar si un trozo de texto T se vincula con otro llamado la hipótesis 
#La relación entre texto e hipótesis es si el texto da suficientes pruebas como para que la hipótesis dada sea cierta
#Puede ser una tarea de clasificacion en la que la etiqueta es True/False
#Se consiguen buenos resultados simplemente con la similitud entre texto e hipótesis a nivel de palabra
#La información de la hipótesis deberá estar en el texto pero no al revés
#Método hyp_extra() determina  esto
def rte_features(rtepair):
    extractor = nltk.RTEFeatureExtractor(rtepair)
    features = {}
    features['word_overlap'] = len(extractor.overlap('word'))
    features['word_hyp_extra'] = len(extractor.hyp_extra('word'))
    features['ne_overlap'] = len(extractor.overlap('ne'))
    features['ne_hyp_extra'] = len(extractor.hyp_extra('ne'))
    return features
#rtepair=nltk.corpus.rte.pairs(['rte3_dev.xml'])[33]
#extractor=nltk.RTEFeatureExtractor(rtepair)
#print(extractor.text_words)
#print(extractor.hyp_words)
#print(extractor.overlap('word'))
#print(extractor.overlap('ne'))
#print(extractor.overlap('word'))
"""

#3 Evaluacion

#3.1 El conjunto Test
#Importante que el conjunto de entrenamiento sea distinto al de test porque si no no está funcionando y simplemente reconoce los datos
#Según que tarea puede ser necesario un conjunto de entrenamiento más grande o más pequeño 
#Si el numero de etiquetas es muy alto se debe tener un conjunto de tal modo que la etiqueta menos frecuente aparezca al menos 50 veces
#Hay que tener en cuenta también el grado de similitud entre el conjunto test y el de desarrollo
#Cuanto más se parezcan más dificil será la generalización 
#En el ejemplo creamos un conjunto de entrenamiento de forma aleatoria de una fuente
import random
from nltk.corpus import brown
"""
tagged_sents = list(brown.tagged_sents(categories='news'))
random.shuffle(tagged_sents)
size = int(len(tagged_sents) * 0.1) 
train_set, test_set = tagged_sents[size:], tagged_sents[:size] 
#En este caso los dos conjuntos son del mismo genero y no sabemos si se puede generalizar a otros 
#De hecho al usar random estamos mezclando los dos conjuntos, cosa que no es recomendable
#En este otro ejemplo nos aseguramos de tener diversidad de fuentes
file_ids = brown.fileids(categories='news')
size = int(len(file_ids) * 0.1)
train_set = brown.tagged_sents(file_ids[size:])
test_set = brown.tagged_sents(file_ids[:size])
#Incluso mejor es usar cosas de documentos más distintos
train_set = brown.tagged_sents(categories='news')
test_set = brown.tagged_sents(categories='fiction')
"""
#3.2 Precisión
#classifier=nltk.NaiveBayesClassifier.train(train_set)
#print('Accuracy: {:4.2f}'.format(nltk.classify.accuracy(classifier, test_set)))
#Es importante ver la precision dentro de las frecuencias de aparicion de diferentes palabras 

#3.3 Precision y retirada
#la precisión en una labor de búsqueda es bastante irrelevante ya que el numero de documentos irrelevantes es bastante superior a los relevantes y estaría cerca del 0%
#Para estas tareas se definen otras medidas de éxito como el verdadero positivo, el verdadero negativo, el falso positivo(o error tipo 1) y el falso negativo(error tipo 2)
#Se definen nuevas métricas, precision (TP/(TP+FP))
#"Recall" (TP/TP+FN)
#F-measure (2*Precision*Recall)/(Precision+Recall)

#3.4 Matrrices de Confusión
#Cuando se trabaja con tres o más etiquetas interesa subdividir los errores que comete el modelo basado en los tipos de errores que se cometen
#La matriz de confusión es una tabla en la que cada celda indica cuando una etiqueta j se predijo cuando la correcta era i, las entradas en diagonal son predicciones correctas
"""
def tag_list(tagged_sents):
    return [tag for sent in tagged_sents for (word,tag) in sent]
def apply_tagger(tagger,corpus):
    return [tagger.tag(nltk.tag.untag(sent)) for sent in corpus]
gold = tag_list(brown.tagged_sents(categories='editorial'))
test = tag_list(apply_tagger(t2, brown.tagged_sents(categories='editorial')))
cm=nltk.ConfusionMatrix(gold,test)
print(cm.pretty_format(sort_by_count=True, show_percents=True,truncate=9))
"""

#3.5 Cross-Validation
#Si hacemos el conjunto de entrenamiento pequeño para tener un conjunto de test grande puede ser malo también
#Cross-validation consiste en hacer evaluaciones sobre conjuntos de test muy pequeños y combinar las puntuaciones 
#Nos permite examinar como rinde el modelo con diferentes conjuntos de datos


#4 Arboles de decisión
#Son diagramas de flujo que seleccionan etiquetas para valores de entrada
#Esta formado por nodos de decision que comprueban las características y nodos hoja que asignan etiquets
#Se comienza desde el nodo raíz
#Un tocon de decisión es un arbol de decision con un solo nodo que decide como clasificar entradas basandose en una única característica, tiene una hoja por cada valor posible
#Se suele asignar en los valores más frecuentes en el conjunto de entrenamiento
#Para construir el arbol de decisión se elije un tocon de decision y se comprueba la precision 
#Las hojas que no tienen suficiente precisión se reemplazan con nuevos tocones de decisión.
 
#4.1 Entropía y obtención de información
#Otro metodo de eleccion de la característica más informativa es information gain  que mide como de organizado están los valores de entrada cuando se los divide por una característica
#Esto se hace calculando la entropía de las etiquetas que aumenta con la variedad de input  
#Entropía es la suma de la probabilidad de cada etiqueta por el logaritmo de su misma probabilidad
import math
def entropy(labels):
    freqdist = nltk.FreqDist(labels)
    probs = [freqdist.freq(l) for l in freqdist]
    return -sum(p * math.log(p,2) for p in probs)
print(entropy(['male', 'male', 'male', 'male']))
print(entropy(['male', 'female', 'male', 'male']))
print(entropy(['female', 'male', 'female', 'male']))
print(entropy(['female', 'female', 'male', 'female']))
print(entropy(['female', 'female', 'female', 'female'])) 
#El problema de la entropia como elemento es que las medidas de entropía a medida que vas avanzando por los nodos son peores porque tienen menos info con la que trabajar
#otro problema de esto es que fuerza la subordinación de características a la jerarquía del árbol 
#si no se hiciera esto el numero de ramas crecería de forma exponencial y no sería viable
#Pero por el otro lado limita la capacidad de explotar las características


#5 Naive Bayes Classifiers
#Con estos clasificadores se tienen en cuenta todas las características
#Para escoger una etiqueta para una entrada el clasificador calcula la probabilidad a priori para cada etiqueta compribando la fecuencia de aparicion de cada palabra 
#Cada característica se combina con la probabilidad a priori y se llega al estimado de cada etiqueta, la más alta se asigna
#Las características individuales "votan en contra" de las etiquetas reduciendo la probabilidad de estas con la probabilidad condicionada de la caracteristica y la etiqueta

#5.1 Modelo probabilistico subyacente
#Está la asunmption de que cada valor de entrada se genera eligiendo primero una etiqueta de clase para ese valor y luego se genera cada característica de forma independiente a las otras
#Cuando en verdad las características suelen estar relacionadas 
#Se conoce como la assumption de navie Bayes que hace más sencillo combinar las contribuciones de diferentes caracteristicas al no tener que preocuparse por las interacciones
#Con esto se puede calcular la probabilidad condicionada de la etiqueta dado un conjunto de caracteristicas
#Si se quiere generar un estimado de la probabilidad para cada etiqueta se computa la probabilidad de todas las caracteristicas

#5.2 Zero Counts and Smoothing
#P(f|label)=count(f,label)/count(label)
#Es problemático cuando una característica nunca ocurre con una etiqueta ya que esta nunca se asignará
#Que no hayamos visto una caracteristica/etiqueta no significa que no pueda ocurrir y este modelo elimina la posibilidad
#No es fiable cuando count(f) es bajo 
#Se suelen usar tecnicas de "smoothing" patra calcular P(f|label)
#La estimacion de semejanza esperada para la probabilidad de una característica añade 0.5 a count(f|label) 
#La estimacion de Heldout utiliza el heldout corpus para calcular la relacion entre frecuencias y probabilidades
#nltk.probability da soporte a muchas técnicas de smoothing
""" 
5.3 Características no binarias
Se asume que las características se tienen o no(binarias), muchas etiquetas se pueden cambiar por valores binarios rojo -> es_rojo
lo mismo con valores numericos 5 -> 4<x<6
También se pueden usar métodos de regresión para modelar las probabilidades en características numéricas 
"""
"""
5.4 La inociecia de la independencia
Si en un clasificador de Bayes se tuviera en cuenta la dependencia se contaría el doble el efecto de las características correlacionadas creando cierta ponderación.
Esto ocurre porque la contribución de la etiqueta se suma a la de la dependiente aumentando la probabilidad y dando más peso del que se merece
"""
"""
5.5 La causa del doble conteo
Se puede solucionar teniendo en cuenta las interacciones entre contribuciones de etiqueta durante el entrenamiento en vez de durante la clasificación
y con esto ajustar las contribuciones de las características individuales
P(features,label)=w[label] * Prod f|in|features w[f,label]
w[label] aquí es la puntuación inicial de la etiqueta dada y 
w[f,label] es la contribución de otra característica a una etiqueta
se les llama los pesos del modelo y usando algoritmo de Bayes les damos valores por separado
w[label]=P(label)
w[f,label]=P(f|label)
"""

"""
6. Clasificadores de Máxima Entropía
Es un modelo que se parece al de bayes pero en vez de usar las probabilidades de parámetro usa técnicas de búsqueda para buscar un conjunto de parámetros que permiten maximizar el rendimiento
El "total likelihood" se define como:
P(features)=Sumatorio x|in|corpus(P(label(x)|features(x)))
P(label|features)=P(label,features)/Sumatorio label(P(label,features))
Para escoger estos modelos utilizan técnicas de optimización iterativa que inicializan los parámetros a valores aleatorios 
y van refinando esos parámetros para llegar a la solución óptima, algunas de estas tecnicas son más eficientes y otras más rápidas
"""
"""
6.1 El modelo de la máxima entropía
Es una generalización del modelo de naive Bayes.
Calcula la posibilidad de cada etiqueta dado un input multiplicando los parámetros aplicables
Este modelo deja al usuario decidir que combinaciones de etiquetas y características reciben sus propios parámetros
Es posible usar un único parámetro para asociar una característica a mas de una etiqueta o por el otro lado mas de una característica a una etiqueta.
Cada combinación de características y etiquetas se llama joint-feature, son propiedades de valores etiquetados. 
Simples caracteristicas son valures no etiquetados
"""
"""
6.2 Maximizando Entropía
La mázima entropia viene motivada por que cada clasificación tenga un modelo que capture las frecuencias de las joint-features individuales sin asunciones. 
Si no se sabe nada a priori se asigna de forma uniforme la probabilidad para que la entropía sea mayor.
Principio de la máxima entropía: Entre las distribuciones que son consistentes con lo que sabemos debemos elegir aquella que tenga mayor entropía.
"""
"""
6.3 Clasificadores Generativos vs Condicionales
El clasificador naive Bayes es un ejemplo de generativo y construye un modelo que predice P(input,label), la probabilidad conjunta de input y label
1¿Cual es la etiqueta más posible para un input?
2¿Como de probable es que una etiqueta dada sea para un input?
3¿Cual es el input más probable?
4¿Como de probable es un valor de input dado?
5¿Como de probable es que haya un valor dado con una etiqueta dada?
6¿Cual es la etiqueta más probable para un input que tengan uno de dos valores?
Por otro lado el clasificador de la máxima entropía es condicional 
predicen P(label|input) la probabilidad de una etiqueta teniendo un input, se usan para las dos primeras preguntas 
Los modelos generaticos son más poderosos ya que calculan la probabilidad condicional en vez de la conjunta.
Por otro lado tienen más parámetros que hay que gestionar y el tamaño del conjunto de entrenamiento es fijo
"""





 