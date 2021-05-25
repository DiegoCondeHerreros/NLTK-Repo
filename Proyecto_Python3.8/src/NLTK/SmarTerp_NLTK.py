# -*- coding: utf-8 -*-
'''
Created on 24 may. 2021

@author: DIEGO
'''
import nltk
from nltk.metrics.scores import accuracy, precision
from nltk.metrics.scores import recall
from nltk.metrics.scores import f_measure
line_raw=[]
filehandle = open("SmarTerp/Rondeau_Seminars_Orthodontics_Level_II_Sample_Case_-_Mai_Part_1.trs", 'r')
#Parseamos excel
while True:
    # read a single line
    line = filehandle.readline()
    split_line=line.split()
    if not line:
        break
    if(not (line[0][0]=="<")):
        if (not (line[0][0]=="@")):
            line_raw.append(split_line)
# close the pointer to that file
filehandle.close()
#Convertimos esta lista de listas en una única lista
raw_annotations_test = []
for sublist in line_raw:
    for item in sublist:
        raw_annotations_test.append(item)    
#raw_annotations_test="este es un (test) hecho en Jamaica"
split_annotations=raw_annotations_test
#Agrupamos los datos de entidades nombradas en tuplas
def group(lst,n):
    for i in range(0,len(lst),n):
        val = lst[i:i+n]
        if len(val) == n:
            yield tuple(val)           
#Partimos del texto en bruto con los paréntesis y con las EN con mayúsculas
tags=[]
#Bucle en el que eliminamos todos los paréntesis del texto
for n,i in enumerate(split_annotations):
    split_annotations[n] = split_annotations[n].replace('(', '')
    split_annotations[n] = split_annotations[n].replace(')', '')
#Bucle que etiqueta los elementos si está en mayúscula como EN y si no como O 
for n,i in enumerate(split_annotations):
    if ((i[0].isupper())  and (split_annotations[n-1][0].islower())):
        tags.insert(n,"B-MISC")
    elif ((i[0].isupper())  and (split_annotations[n-1][0].isupper())):
        tags.insert(n,"I-MISC")
    else:
        tags.insert(n,"O")           
#Ponemos las palabras en mayúscula en minúscula para la predicción
for n,i in enumerate(split_annotations):
    split_annotations[n]=split_annotations[n].lower()
#Creamos texto anotado para la referencia
reference_annotations=[]
for n,i in enumerate(split_annotations):
    reference_annotations.append(i)
    reference_annotations.append(tags[n])
#Ponemos las referencias como tuplas      
reference_annotations = list(group(reference_annotations,2))



print(split_annotations)
raw=""
raw1=""
#Guardamos el documento en bruto y el documento anotado en diferentes ficheros para hacer las pruebas aquí y en spacy
f = open("SmarTerp/Rondeau_Raw.txt", "w")
raw = " ".join(split_annotations)
f.write(raw)
f.close()
f = open("SmarTerp/Rondeau_Reference.txt","w")
f.write('\n'.join('%s %s' % x for x in reference_annotations))
f.close()








#Obtenemos etiquetado POS
tagged_words= nltk.pos_tag(split_annotations)
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
#Convertimos todas las EN en MISC para ajustarnos a SmarTerp.    
for n,i in enumerate(listed_ne):
    if listed_ne[n].startswith("B-"):
        listed_ne[n]="B-MISC"
    if listed_ne[n].startswith("I-"):
        listed_ne[n]="I-MISC"
#Paso a tuplas
nltk_formatted_prediction=list(group(listed_ne,2))
for n,i in enumerate(nltk_formatted_prediction):
    if(i[1]=="B-MISC"):
        print(i)
    if(i[1]=="I-MISC"):
        print(i)
#Da resultados muy buenos, 94.50, 93.73, 94.50 pero no son datos significativos porque al imprimir por pantalla las entidades nombradas
#lo pone todo como O, es razonable pensar que al no estar las palabras en mayúsculas el clasificador no las reconoce siquiera como EN.         
        
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