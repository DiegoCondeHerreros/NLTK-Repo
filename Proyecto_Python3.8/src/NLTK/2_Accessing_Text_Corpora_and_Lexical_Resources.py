# -*- coding: utf-8 -*-
'''
Created on 16 feb. 2021

@author: DIEGO
'''
import nltk
from nltk.book import *

# Un corpus es un texto compuesto de otros textos 
# Con el siguiente mandato obtenemos los nombres de cada texto al que podemos acceder
# nltk.corpus.gutenberg.fileids()
# emma = nltk.corpus.gutenberg.words('austen-emma.txt')
# len(emma)
# Se debe hacer nltk.Text para poder operar sobre los textos del corpus
# emma = nltk.Text(nltk.corpus.gutenberg.words('austen-emma.txt'))
# emma.concordance("surprize")

# Para facilitar las cosas
# from nltk.corpus import gutenberg
# gutenberg.fileids()

# Este programa muestera la longitud de palabra media, la longitud de frase media y el numero medio de apariciones de una palabra
# La funcion raw() nos permite obtener el texto sin ningún procesado ni partición en tokens
# La funcion words() obtiene el texto dividido en palabras
# La funcion sents() obtiene el texto dividido en oraciones, siendo estas una lista de palabras
#for fileid in gutenberg.fileids():
#    num_chars = len(gutenberg.raw(fileid))
#    num_words = len(gutenberg.words(fileid))
#    num_sents = len(gutenberg.sents(fileid))
#    num_vocab = len(set(w.lower() for w in gutenberg.words(fileid)))
#    print(round(num_chars/num_words), round(num_words/num_sents), round(num_words/num_vocab), fileid)
    
# Obtenemos de Macbeth las oraciones con mayor longitud     
#macbeth_sentences = gutenberg.sents('shakespeare-macbeth.txt')
#macbeth_sentences[116]
#longest_length = max(len(s) for s in macbeth_sentences)
#[s for s in macbeth_sentences if len(s) == longest_length]

# Con esto se accede a BD de mensajes de salas de chat
#from nltk.corpus import nps_chat
#chatroom = nps_chat.posts('10-19-20s_706posts.xml')
#chatroom[123]

#Brown es un corpus de más de 500 fuentes, aquí accedemos a las mismas de diversas formas
#Brown es muy util para hayar diferencias estilísticas entre diversos géneros
from nltk.corpus import brown
#print(brown.categories())
#print(brown.words(categories='news'))
#print(brown.words(fileids=['cg22']))
#print(brown.sents(categories=['news', 'editorial', 'reviews']))    

#news_text = brown.words(categories='news')
#fdist = nltk.FreqDist(w.lower() for w in news_text)
#modals = ['can', 'could', 'may', 'might', 'must', 'will']
#for m in modals:
#    print(m + ':', fdist[m], end=' ')
    
#news_text = brown.words(categories='romance')
#fdist = nltk.FreqDist(w.lower() for w in news_text)
#modals = ['what', 'when', 'where', 'who', 'why']
#for m in modals:
#    print(m + ':', fdist[m], end=' ')

#cfd = nltk.ConditionalFreqDist(
#           (genre, word)
#           for genre in brown.categories()
#           for word in brown.words(categories=genre))
#genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']
#modals = ['can', 'could', 'may', 'might', 'must', 'will']
#cfd.tabulate(conditions=genres, samples=modals)
    
# Reuters es un corpus que está dividio en testing y algorithms para indicar el uso de los mismos
from nltk.corpus import reuters
from unicodedata import category
reuters.fileids()
reuters.categories()
#Estos documentos tienen muchas categiorías mientras que en otros corpus solo tienen una 
reuters.categories('training/9865')
reuters.categories(['training/9865','training/9880'])
reuters.fileids('barley')
reuters.fileids(['barley','corn'])
reuters.words('training/9865')[:14]
reuters.words(['training/9865','training/9880'])
reuters.words(categories='barley')
reuters.words(categories=['barley','corn'])

#Inaugural address corpus, está formado por el año y el presidente, extrayendo los 4 primeros caracteres nos quedamos con el año
from nltk.corpus import inaugural
inaugural.fileids()
[fileid[:4] for fileid in inaugural.fileids()]

# Con esto comparamos en una gráfica el uso de 'America' y 'citizen' en el corpus inagural en los últimos siglos
cfd = nltk.ConditionalFreqDist(
           (target, fileid[:4])
           for fileid in inaugural.fileids()
           for w in inaugural.words(fileid)
           for target in ['america', 'citizen']
           if w.lower().startswith(target))
#cfd.plot()

#Hay corpora de muchos idiomas con diferentes codificaciones
#nltk.corpus.cess_esp.words()
#nltk.corpus.floresta.words()
#nltk.corpus.indian.words('hindi.pos')
#nltk.corpus.udhr.fileids()
#nltk.corpus.udhr.words('Javanese-Latin1')[11:]

#from nltk.corpus import udhr
#languages = ['Chickasaw', 'English', 'German_Deutsch',
#     'Greenlandic_Inuktikut', 'Hungarian_Magyar', 'Ibibio_Efik']
#cfd = nltk.ConditionalFreqDist(
#        (lang, len(word))
#        for lang in languages
#        for word in udhr.words(lang + '-Latin1'))
#cfd.plot(cumulative=True)

#raw_text = udhr.raw(Language-Latin1)
#nltk.FreqDist(raw_text).plot().

# CHULETA CORPUS
# fileids()    the files of the corpus
# fileids([categories])    the files of the corpus corresponding to these categories
# categories()    the categories of the corpus
# categories([fileids])    the categories of the corpus corresponding to these files
# raw()    the raw content of the corpus
# raw(fileids=[f1,f2,f3])    the raw content of the specified files
# raw(categories=[c1,c2])    the raw content of the specified categories
# words()    the words of the whole corpus
# words(fileids=[f1,f2,f3])    the words of the specified fileids
# words(categories=[c1,c2])    the words of the specified categories
# sents()    the sentences of the whole corpus
# sents(fileids=[f1,f2,f3])    the sentences of the specified fileids
# sents(categories=[c1,c2])    the sentences of the specified categories
# abspath(fileid)    the location of the given file on disk
# encoding(fileid)    the encoding of the file (if known)
# open(fileid)    open a stream for reading the given corpus file
# root    if the path to the root of locally installed corpus
# readme()    the contents of the README file of the corpus

raw = gutenberg.raw("burgess-busterbrown.txt")
raw[1:20]
words = gutenberg.words("burgess-busterbrown.txt")
words[1:20]
sents = gutenberg.sents("burgess-busterbrown.txt")
sents[1:20]

#Como subir un corpus propio
# 1 Indicas el directorio en el que se encuentra (corpus_root)
# 2 Puedes indicar en el segundo parámetro los ficheros que tiene que leer o * para toodos
#from nltk.corpus import PlaintextCorpusReader
#corpus_root = '/usr/share/dict' 
#wordlists = PlaintextCorpusReader(corpus_root, '.*')
#wordlists.fileids()
#wordlists.words('connectives')

#Ejemplo para subir un corpus existente
#from nltk.corpus import BracketParseCorpusReader
#corpus_root = r"C:\corpora\penntreebank\parsed\mrg\wsj"
#file_pattern = r".*/wsj_.*\.mrg" 
#ptb = BracketParseCorpusReader(corpus_root, file_pattern)
#ptb.fileids()
#len(ptb.sents())
#ptb.sents(fileids='20/wsj_2013.mrg')[19]

#La frecuencia distribuida condicional nos permite ver el numero de ocurrencias por un elemento que cumple una condicion
# En vez de procesar las palabras solas se procesan en parejas
cfd = nltk.ConditionalFreqDist(
          (genre, word)
          for genre in brown.categories()
          for word in brown.words(categories=genre))
#Por cada genero de los dos que hemos elegido vemos todas las palabras asignadas al mismo 
genre_word = [(genre, word)
              for genre in ['news', 'romance']
              for word in brown.words(categories=genre)]
len(genre_word)
#Creamos la frecuencia condicional distribuida y comprobamos condiciones
cfd = nltk.ConditionalFreqDist(genre_word)
cfd.conditions()
print(cfd['news'])
print(cfd['romance'])
print(cfd['romance'].most_common(20))
print(cfd['romance']['could'])

#En plot y tabular podemos indicar que condiciones queremos que aparezcan en la tabla
from nltk.corpus import udhr
languages = ['Chickasaw', 'English', 'German_Deutsch',
    'Greenlandic_Inuktikut', 'Hungarian_Magyar', 'Ibibio_Efik']
#Linea 1: Indica que la condicion es el nombre y lo que se cuenta deriva de longitudes de palabras
#Linea 2: Recorre los elementos de la lista de lenguajes
#Linea 3: Recorre todas las palabras del corpus udhr cogiendolo de los ficheros que corresponden a esos lenguajes
#Linea 4: En el tabular solo pone los idiomas que quiere poner en condiciones, limita el numero de resultados con samples y 
cfd = nltk.ConditionalFreqDist(  
          (lang, len(word))
          for lang in languages
          for word in udhr.words(lang + '-Latin1'))
cfd.tabulate(conditions=['English', 'German_Deutsch'],
             samples=range(10), cumulative=True)

#Calculamos que dias de la semana son más frecuentes en los géneros romance y noticias
days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
genre = ['romance','news']
cfd = nltk.ConditionalFreqDist(
    (genre,word)
    for day in days
    for genre in ['news','romance'] 
    for word in brown.words(categories=genre))
cfd.tabulate(samples=days)

#El método bigrams genera un conjunto de bigramas aleatorios
sent = ['In','the','beginning','God','created','the','heaven','and','the','earth','.']
list(nltk.bigrams(sent))
#Generate model se usa para crear texto aleatorio
def generate_model(cfdist, word, num=15):
    for i in range(num):
        print(word, end=' ')
        word = cfdist[word].max()

text = nltk.corpus.genesis.words('english-kjv.txt')
bigrams = nltk.bigrams(text)
cfd = nltk.ConditionalFreqDist(bigrams)
print(cfd['living'])
generate_model(cfd,'living')

#CHULETA DE METODOS DE DISTRIBUCION DE FRECUENCIA CONDICIONAL
# cfdist = ConditionalFreqDist(pairs)    create a conditional frequency distribution from a list of pairs
# cfdist.conditions()    the conditions
# cfdist[condition]    the frequency distribution for this condition
# cfdist[condition][sample]    frequency for the given sample for this condition
# cfdist.tabulate()    tabulate the conditional frequency distribution
# cfdist.tabulate(samples, conditions)    tabulation limited to the specified samples and conditions
# cfdist.plot()    graphical plot of the conditional frequency distribution
# cfdist.plot(samples, conditions)    graphical plot limited to the specified samples and conditions
# cfdist1 < cfdist2    test if samples in cfdist1 occur less frequently than in cfdist2

#Wordlist corpora es un es una lista de palabras que se usa para comprobar errores léxicos
def unusual_words(text):
    text_vocab = set(w.lower() for w in text if w.isalpha())
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    unusual = text_vocab - english_vocab
    return sorted(unusual)
unusual_words(nltk.corpus.gutenberg.words('austen-sense.txt'))
unusual_words(nltk.corpus.nps_chat.words())
# Las stopwords son palabras que aparecen mucho en un texto
from nltk.corpus import stopwords
stopwords.words('english')

#Fracción de palabras del texto que no está en el stopwords
def content_fraction(text):
    stopwords = nltk.corpus.stopwords.words('english')
    content = [w for w in text if w.lower() not in stopwords]
    return len(content) / len(text)
content_fraction(nltk.corpus.reuters.words())

#Para solucionar puzzle
#Linea 1 del for para limitar la longitud de la palabra
#Linea 2 del for para obligar que la palabra contenga la w
#Linea 3 del for para hacer la frecuencia distribuida de cada una de las soluciones 

puzzle_letters = nltk.FreqDist('egivrvonl')
obligatory = 'r'
wordlist = nltk.corpus.words.words()
[w for w in wordlist if len(w) >= 6 
                     and obligatory in w 
                     and nltk.FreqDist(w) <= puzzle_letters]

# Esta dividido por genero y este texto calcula que palabras son iguales para ambos géneros
names = nltk.corpus.names
names.fileids()
['female.txt', 'male.txt']
male_names = names.words('male.txt')
female_names = names.words('female.txt')
[w for w in male_names if w in female_names]

#Se comprueba que nombres acaban en a y son femeninos
cfd = nltk.ConditionalFreqDist(
          (fileid, name[-1])
          for fileid in names.fileids()
          for name in names.words(fileid))
cfd.plot()

#Diccionario de pronunciación de las palabras, se usa para sintetizadores, carga los elementos y te imprime algunos
entries = nltk.corpus.cmudict.entries()
len(entries)
for entry in entries[42371:42379]:
    print(entry)

#Otra manera de recorrer la tabla
#Word y pron son la palabra y la pronunciacion que busca en la tabla
#Se busca palabras que tengan hasta 3 fonemas 
# Si cumple la condicion lo asigna a pron
for word, pron in entries: 
    if len(pron) == 3: 
        ph1, ph2, ph3 = pron 
        if ph1 == 'P' and ph3 == 'T':
            print(word, ph2, end=' ')
#For que busca palabras que terminen con estos 4 fonemas
syllable = ['N', 'IH0', 'K', 'S']
[word for word, pron in entries if pron[-4:] == syllable]

#Este bucle te busca palabras que acaben con mn
[w for w, pron in entries if pron[-1] == 'M' and w[-1] == 'n']
# Este bucle busca aquellos fonemas que suenan como la n aunque no se escriba con la misma
sorted(set(w[:2] for w, pron in entries if pron[0] == 'N' and w[0] != 'n'))

#Funcion que te busca donde se hace stress en las palabras
def stress(pron):
    return [char for phone in pron for char in phone if char.isdigit()]
#Búsqueda de palabras que se ajusten a un patron de pronunciación
[w for w, pron in entries if stress(pron) == ['0', '1', '0', '2', '0']]
[w for w, pron in entries if stress(pron) == ['0', '2', '0', '1', '0']]

#Se puede acceder también mediante la estructura de datos diccionario de python
prondict = nltk.corpus.cmudict.dict()
prondict['fire']
text = ['natural', 'language', 'processing']
[ph for w in text for ph in prondict[w][0]]

#Swadesh es una lista de palabras comparativa
from nltk.corpus import swadesh
swadesh.fileids()
swadesh.words('en')
#Se puede acceder también con el método entries(), especificando una lista de lenguajes
fr2en = swadesh.entries(['fr','en'])
print(fr2en)
translate = dict(fr2en)
translate('chien')
translate('jeter')

# Esto mismo se puede hacer para abarcar varios lenguajes
de2en = swadesh.entries(['de', 'en'])    # German-English
es2en = swadesh.entries(['es', 'en'])    # Spanish-English
translate.update(dict(de2en))
translate.update(dict(es2en))
translate['Hund']
translate['perro']

#ShoeBox y ToolBox son conjuntos de herramientas que consisten en un conjunto de entradas
# Este ejemplo es el idioma Rotokas, cada par es o bien la palabra y su tipo o la palabra y su traduccion al inglés
#from nltk.corpus import toolbox
#toolbox.entries('rotokas.dic')

#Wordnet nos permite acceder a sinónimos y antónimos, synset es conjunto de sinonimos
from nltk.corpus import wordnet as wn
#Accedemos al conjunto de sinónimos de una palabra, los nombres que lo componen, la definicion y algunos ejemplos
#wn.synsets('motorcar')
#wn.synset('car.n.01').lemma_names()
#wn.synset('car.n.01').definition()
#wn.synset('car.n.01').examples()
#Un lemma es la union de una palabra y su sinónimo
#Accedemos a todos los lemamas de un synset
#Accedemos a un lemma particular
#Accedemos al synset correspondiente a un lemma
#Accedemos al nombre del lemma
#wn.synset('car.n.01').lemmas()
#wn.lemma('car.n.01.automobile')
#wn.lemma('car.n.01.automobile').synset()
#wn.lemma('car.n.01.automobile').name()

#Hay palabras concretas que solo tienen un synset y otras que son ambiguas y tienen varios
#wn.synsets('car')
#for synset in wn.synsets('car'):
#    print(synset.lemma_names())
#wn.lemmas('car')

#Consultamos los hipónimos de una palabra
#motorcar = wn.synset('car.n.01')
#types_of_motorcar = motorcar.hyponyms()
#types_of_motorcar[0]
#sorted(lemma.name() for synset in types_of_motorcar for lemma in synset.lemmas())
#Hiperónimos
#wn.synset('tree.n.01').part_meronyms()
#wn.synset('tree.n.01').substance_meronyms()
#wn.synset('tree.n.01').member_holonyms()
#Relacion de entailment
#wn.synset('walk.v.01').entailments()
#Relación de antonimia
#wn.lemma('supply.n.02.supply').antonyms()



