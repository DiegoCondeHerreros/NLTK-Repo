# -*- coding: utf-8 -*-
'''
Created on 22 abr. 2021

@author: DIEGO
'''
import nltk
from nltk.metrics.scores import accuracy
from nltk.metrics.scores import recall
from nltk.metrics.scores import f_measure
#Cargamos los datos de wikigold.
raw_annotations= open("wikigold.conll.txt",encoding="UTF-8").read()
split_annotations=raw_annotations.split()
#Separamos las etiquetas para ajustarnos a los experimentos del paper
for n,i in enumerate(split_annotations):
    if i=="I-PER":
        split_annotations[n] = "PERSON"
    if i=="I-ORG":
        split_annotations[n] = "ORGANIZATION"
    if i=="I-LOC":
        split_annotations[n] = "LOCATION"

#Agrupamos los datos de entidades nombradas en tuplas
def group(lst,n):
    for i in range(0,len(lst),n):
        val = lst[i:i+n]
        if len(val) == n:
            yield tuple(val)
            
reference_annotations = list(group(split_annotations,2)) 

#Limpiamos los datos para usarlo con el clasificador NER
pure_tokens= split_annotations[::2]

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

#Para mantener consistencia con las anotaciones de clase
for n,i in enumerate(listed_ne):
    if i=="B-PERSON":
        listed_ne[n]="PERSON"
    if i=="I-PERSON":
        listed_ne[n]="PERSON"
    if i=="B-ORGANIZATION":
        listed_ne[n]="ORGANIZATION"
    if i=="I-ORGANIZATION":
        listed_ne[n]="ORGANIZATION"
    if i=="B-LOCATION":
        listed_ne[n]="LOCATION"
    if i=="I-LOCATION":
        listed_ne[n]="LOCATION"
    if i=="B-GPE":
        listed_ne[n]="LOCATION"
    if i=="I-GPE":
        listed_ne[n]="LOCATION"
    
#Paso a tuplas
nltk_formatted_prediction=list(group(listed_ne,2))

#Obtenemos la precisi�n con el m�todo accuracy de nltk.metrics.scores
nltk_accuracy=accuracy(reference_annotations, nltk_formatted_prediction)
nltk_recall=recall(set(reference_annotations), set(nltk_formatted_prediction))
nltk_f=f_measure(set(reference_annotations), set(nltk_formatted_prediction),1)
print(nltk_accuracy)    
print(nltk_recall)
print(nltk_f)


           
 