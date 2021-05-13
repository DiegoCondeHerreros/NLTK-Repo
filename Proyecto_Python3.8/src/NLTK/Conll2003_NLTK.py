# -*- coding: utf-8 -*-
'''
Created on 6 may. 2021

@author: DIEGO
'''
import nltk
from nltk.metrics.scores import accuracy
from nltk.metrics.scores import recall
from nltk.metrics.scores import f_measure
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
 
#Paso a tuplas
nltk_formatted_prediction=list(group(listed_ne,2))
#Limpiamos las anotaciones de referencia para que solo contenga las etiquetas de NER.
anotacion=[]
for tuplas in reference_annotations :
    anotacion.append(tuplas[0]) 
    anotacion.append(tuplas[3])
print(anotacion)
reference_annotations=list(group(anotacion, 2))
#Calculamos las distintas puntuaciones.
nltk_accuracy=accuracy(reference_annotations, nltk_formatted_prediction)
nltk_recall=recall(set(reference_annotations), set(nltk_formatted_prediction))
nltk_f=f_measure(set(reference_annotations), set(nltk_formatted_prediction),1)
print(nltk_accuracy)    
print(nltk_recall)
print(nltk_f)