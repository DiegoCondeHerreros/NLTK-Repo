# -*- coding: utf-8 -*-
'''
Created on 14 feb. 2021

@author: DIEGO
'''
import nltk 
#sentence = """At eight o' clock on Thursday morning Arthur didn't feel very good."""
#tokens = nltk.word_tokenize(sentence)
#print(tokens)
#tagged = nltk.pos_tag(tokens)
#print(tagged[0:6])
#entities = nltk.chunk.ne_chunk(tagged)
#print(entities)
#nltk.download()
from nltk.book import *
text1
text2

# Metodo concordance te busca todas las instancias de una palabra en un texto y un poco de contexto
#print(text1.concordance('monstrous'))
#print(text2.concordance('affection'))
#print(text3.concordance('lived'))
#print(text4.concordance('nation'))

# Metodo similar te permite encontrar palabras que suelen aparecer en el mismo contexto que la indicada
#print(text1.similar('monstrous'))
#print(text2.similar('monstrous'))

#Metodo context te permite encontrar contextos comunes a dos o mas palabras en un texto
#print(text2.common_contexts(['monstrous','very']))

# Prueba de esto mismo con palabra distinta y contexto distinto
#print(text2.similar('father'))
#print(text3.similar('father'))
#print(text2.common_contexts(['brother','wife']))

#Metodo dispersion_plot permite ver la frecuencia de ocurrencias de  determinadas palabras a lo largo del texto
import matplotlib
#text4.dispersion_plot(['citizens', 'democracy', 'freedom', 'duties', 'America', 'liberty', 'constitution']) 

#Metodo generate genera un texto aleatorio del estilo de un texto dado
#text3.generate()

#Al poner text3 como un set lo que hace es poner juntas las diferentes instancias de las mismas palabras como una sola, de esta manera contamos el numero de palabras distintas 
len(set(text3))
# Riqueza léxica es el cociente entre el numero de palabras distintas entre todas las palabras del texto
riqueza_lexica = len(set(text3))/len(text3)
#Porcentaje del texto que ocupa una palabra
perce_word = 100*text5.count('lol')/len(text5)
#print(text5.count('lol'))
#print(perce_word)

def lexical_diversity(text):
    return len(set(text))/len(text)
def percentage(count,total):
    return 100*count/total

#print(lexical_diversity(text3))
#print(lexical_diversity(text5))
#print(percentage(text4.count('a'),len(text4)))

#Para obtener la distribución por frecuencias de las palabras de un texto
#fdist1 = FreqDist(text1)
#print(fdist1)
#Las 50 que más aparecen
#fdist1.most_common(50)

#fdist2 = FreqDist(text2)
#print(fdist2)
#print(fdist2.most_common(50))
#Si se quisiera saber las palabras que ocurren con menor frecuencia se usa este método 
#fdist2.hapaxes()

#De esta manera se obtiene una lista que tiene todas las palabras de 15 caracteres o más
#V = set(text1)
#long_words = [w for w in V if len(w) > 15]
#print(sorted(long_words))

#Encontrar las palabras largas que más aparecen puede ser util para discernir cosas sobre un texto
#fdist5 = FreqDist(text5)
#sorted(w for w in set(text5) if len(w) > 7 and fdist5[w] > 7)
# Collocation es una secuencia de palabras que suelen aparecer juntas, no hay ninguna que sea facilmente sustituible
# Un bigrama es una pareja de palabras
# print(list(bigrams(['more', 'is', 'said', 'than', 'done'])))
# text4.collocations()
# text8.collocations()

#Contamos la cantidad de palabras que hay de determinada longitud
# longitudes = [len(w) for w in text1]
# fdist = FreqDist(len(w) for w in text1)
# print(fdist)

#Más cosas que podemos sacar de la distribución
# fdist.most_common()
# fdist.max()
# fdist[3]
# fdist.freq(3)

# CHULETA DE FUNCIONES Y FÓRMULAS
# fdist = FreqDist(samples)    create a frequency distribution containing the given samples
# fdist[sample] += 1    increment the count for this sample
# fdist['monstrous']    count of the number of times a given sample occurred
# fdist.freq('monstrous')    frequency of a given sample
# fdist.N()    total number of samples
# fdist.most_common(n)    the n most common samples and their frequencies
# for sample in fdist:    iterate over the samples
# fdist.max()    sample with the greatest count
# fdist.tabulate()    tabulate the frequency distribution
# fdist.plot()    graphical plot of the frequency distribution
# fdist.plot(cumulative=True)    cumulative plot of the frequency distribution
# fdist1 |= fdist2    update fdist1 with counts from fdist2
# fdist1 < fdist2    test if samples in fdist1 occur less frequently than in fdist2

#CHULETA DE OPERADORES DE COMPARACIÓN DE PALABRAS
# s.startswith(t)    test if s starts with t
# s.endswith(t)    test if s ends with t
# t in s    test if t is a substring of s
# s.islower()    test if s contains cased characters and all are lowercase
# s.isupper()    test if s contains cased characters and all are uppercase
# s.isalpha()    test if s is non-empty and all characters in s are alphabetic
# s.isalnum()    test if s is non-empty and all characters in s are alphanumeric
# s.isdigit()    test if s is non-empty and all characters in s are digits
# s.istitle()    test if s contains cased characters and is titlecased (i.e. all words in s have initial capitals)

#Acceso a bots de chat primitivos
nltk.chat.chatbots()











