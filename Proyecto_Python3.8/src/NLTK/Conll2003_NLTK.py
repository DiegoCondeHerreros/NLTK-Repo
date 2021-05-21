# -*- coding: utf-8 -*-
'''
Created on 6 may. 2021

@author: DIEGO
'''
import nltk
from nltk.metrics.scores import accuracy, precision
from nltk.metrics.scores import recall
from nltk.metrics.scores import f_measure
from nltk.tag import StanfordNERTagger
from nltk.tag.stanford import StanfordTagger
#Código necesario para que no de un LookupError al ejecutar StanfordNER
import os
java_path = "C:/Program Files/Java/jdk1.8.0_171/bin/java.exe"
os.environ['JAVAHOME'] = java_path
#Cargamos los datos de wikigold.
raw_annotations_test= open("test.txt",encoding="UTF-8").read()
split_annotations=raw_annotations_test.split()

#Agrupamos los datos de entidades nombradas en tuplas
def group(lst,n):
    for i in range(0,len(lst),n):
        val = lst[i:i+n]
        if len(val) == n:
            yield tuple(val)
            
reference_annotations = list(group(split_annotations,4)) 
#Limpiamos los datos para usarlo con el clasificador NER
pure_tokens= split_annotations[::4]
#Obtenemos etiquetado POS
tagged_words= nltk.pos_tag(pure_tokens)
#Obtenemos etiquetado de EN 
nltk_unformatted_prediction = nltk.ne_chunk(tagged_words)
#Como devuelve un arbol lo convertimos en string 
multiline_string = nltk.chunk.tree2conllstr(nltk_unformatted_prediction)
#A su vez convertimos esto en una lista
listed_pos_and_ne= multiline_string.split()
#Borramos las etiquetas POS y renombramos la variable
del listed_pos_and_ne[1::3]
listed_ne=listed_pos_and_ne
#Limpiamos las anotaciones de referencia para que solo contenga las etiquetas de NER.
anotacion=[]
for tuplas in reference_annotations :
    anotacion.append(tuplas[0]) 
    anotacion.append(tuplas[3])
known_entities= ["B-PERSON","I-PERSON","B-ORGANIZATION","I-ORGANIZATION","B-LOCATION","I-LOCATION","B-GPE","I-GPE","O"]
for n,i in enumerate(anotacion):
    if i=="B-PER":
        anotacion[n] = "B-PERSON"
    if i=="I-PER":
        anotacion[n] = "I-PERSON"
    if i=="B-ORG":
        anotacion[n] = "B-ORGANIZATION"
    if i=="I-ORG":
        anotacion[n] = "I-ORGANIZATION"
    if i=="B-LOC":
        anotacion[n] = "B-LOCATION"
    if i=="I-LOC":
        anotacion[n] = "I-LOCATION"
#print(listed_ne)
for n,i in enumerate(listed_ne):
    if i=="B-GPE":
        listed_ne[n]="B-LOCATION"
    if i=="I-GPE":
        listed_ne[n]="I-LOCATION"
    if i not in known_entities:
        if listed_ne[n].startswith("B-"):
            listed_ne[n]="B-MISC"
        if listed_ne[n].startswith("I-"):
            listed_ne[n]="I-MISC"
#print(listed_ne)
reference_annotations=list(group(anotacion, 2))
#print(reference_annotations)
#print(nltk_formatted_prediction)
#Paso a tuplas
nltk_formatted_prediction=list(group(listed_ne,2))
#Calculamos las distintas puntuaciones.

nltk_precision=precision(set(reference_annotations),set(nltk_formatted_prediction))
nltk_accuracy=accuracy(reference_annotations, nltk_formatted_prediction)
nltk_recall=recall(set(reference_annotations), set(nltk_formatted_prediction))
nltk_f=f_measure(set(reference_annotations), set(nltk_formatted_prediction),1)
print("NLTK-Precision")
print(nltk_precision)
print("NLTK-Accuracy")
print(nltk_accuracy)    
print("NLTK-Recall")
print(nltk_recall)
print("NLTK-F1")
print(nltk_f)

#####STANFORD NER TAGGER#######
#Hay tres programas de NER:
#StanfordNER que es la que se emplea en el código a continuación y consiste en el clasificador
#CoreNLP es una pipeline de procesamiento pero a efectos de ner se usa StanfordNER asi que da igual
#StanfordNLP o Stanza 
st = StanfordNERTagger('StanfordNER/english.conll.4class.distsim.crf.ser.gz',
                       'StanfordNER/stanford-ner-4.2.0.jar',
                       encoding='utf-8')                  
stanford_prediction = st.tag(pure_tokens)
#De base esto va a dar puntuaciones malísimas porque StanfordNER no tiene esquema de etiquetado BIO luego da resultados alrededor del 63-64
#Cambiamos las anotaciones de referencia para ignorar esquema de etiquetado IOB
#Con estos cambios da: A: 0.8166, P: 0.6386, R: 0.9237, F:0.6386 pero hasta que punto sirve si no usa IOB?
reference_annotations_aux=reference_annotations
for n,i in enumerate(reference_annotations_aux):
    print(n)
    if i[1]=="B-PERSON":
        reference_annotations_aux[n] = "PERSON"
    if i[1]=="I-PERSON":
        reference_annotations_aux[n] = "PERSON"
    if i[1]=="B-ORGANIZATION":
        reference_annotations_aux[n] = "ORGANIZATION"
    if i[1]=="I-ORGANIZATION":
        reference_annotations_aux[n] = "ORGANIZATION"
    if i[1]=="B-LOCATION":
        reference_annotations_aux[n] = "LOCATION"
    if i[1]=="I-LOCATION":
        reference_annotations_aux[n] = "LOCATION"
#print(reference_annotations_aux)
print(stanford_prediction)
print("StanfordNER-Accuracy")
stanford_accuracy = accuracy(reference_annotations_aux, stanford_prediction)
print(stanford_accuracy)
print("StanfordNER-Precision")
stanford_precision=precision(set(reference_annotations_aux),set(stanford_prediction))
print(stanford_precision)
print("StanfordNER-Recall")
stanford_recall=recall(set(reference_annotations_aux),set(stanford_prediction))
print(stanford_recall)
print("StanfordNER-F1")
stanford_f=f_measure(set(reference_annotations_aux),set(stanford_prediction),1)
print(stanford_f)

######STANZA#########
#Esto no funciona, no me deja descargar el pipeline por algún motivo parece ser que es un problema común al que de momento no he encontrado solución
#import stanza
#nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')
#doc = nlp("Barack Obama was born in Hawaii.")
#print(doc)
#print(doc.entities)


#