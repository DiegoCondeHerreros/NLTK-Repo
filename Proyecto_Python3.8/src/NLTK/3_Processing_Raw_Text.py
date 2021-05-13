# -*- coding: utf-8 -*-
'''
Created on 22 feb. 2021

@author: DIEGO
'''
import nltk, re, pprint
from nltk import word_tokenize

#Se accede a otros tipos de texto, el 2554 es Crime and Punishment
#Se hace un acceso al texto mediante una request
from urllib import request
from test.test_urllib import urlopen
#url = "http://www.gutenberg.org/files/2554/2554-0.txt"
#response = request.urlopen(url)
#raw = response.read().decode('utf8')
#print(type(raw))
#print(len(raw))
#print(raw[:75])
#Este texto en "raw" nos devuelve el propio texto con espacios en blanco, saltos de línea y retornos de carro, nos interesa más partirlos en tokens
#tokens = word_tokenize(raw)
#print(type(tokens))
#print(len(tokens))
#print(tokens[:10])

#Ponemos la lista de tokens en formato nltk.texto 
# Los textos de project gutemberg son collocations y tienen información al principio que hay que filtrar como el autor, la licencia, etc, se tiene que inspeccionar manualmente
#text = nltk.Text(tokens)
#type(text)
#text[1024:1062]
#text.collocations()

# De esta manera nos saltamos la info a filtrar, buscando donde empieza
#raw.find("PART I")
#raw.rfind("End of Project Gutenberg's Crime")
#raw = raw[5338:1157743]
#raw.find("PART I")

# Así obtenemos un HTML, una noticia de la BBC y accediendo al html que devuelve la request. 
#url = "http://news.bbc.co.uk/2/hi/health/2284783.stm"
#html = request.urlopen(url).read().decode('utf8')
#html[:60]

# Para manipular el HTML usaremos una librería que se llama beautiful soup
#TODO: Instalar BeautifulSoup si tengo que currar con HTML
#from bs4 import BeautifulSoup
#raw = BeautifulSoup(html,'html.parser').get_text()
#tokens = word_tokenize(raw)
#print(tokens) 
#tokens = tokens[110:390]
#text = nltk.Text(tokens)
#text.concordance('gene')

#Con la librería Universal Feed Parser se puede acceder al contenido de un blog
#TODO: Instalar Universal Feed Parser por si tengo que acceder a estos recursos
#import feedparser
#llog = feedparser.parse("http://languagelog.ldc.upenn.edu/nll/?feed=atom")
#llog['feed']['title']
#len(llog.entries)
#post = llog.entries[2]
#post.title
#content = post.content[0].value
#content[:70]
#raw = BeautifulSoup(content, 'html.parser').get_text()
#word_tokenize(raw)

#Para leer de un archivo local
#f = open('documento.txt')
#print(f.read())
#De este modo se imprime por líneas y strip elimina los saltos de línea
#f1 = open('documento.txt','rU')
#for line in f1:
#    print(line.strip())

#Para acceder a corpus que estén guardados en local
#path = nltk.data.find('corpora/gutenberg/melville-moby_dick.txt')
#raw = open(path, 'rU').read()

#Para acceder a texto introducido por el usuario
#s = input("Enter some text: ")
#print("You typed", len(word_tokenize(s)), "words.")

#El siguiente es un ejemplo del funcionamiento de la pipeline
#Con los siguientes comandos se descarga uba pagina web, se elimina contenido indeseado y accede a lo que interesa
#Convierte HTML en ASCII
#html = urlopen(url).read
#raw = nltk.clean_html(html)
#raw = raw[750:23506]
#Tokeniza el texto, selecciona tokens de interes y crea texto de NLTK
#Convierte ASCII en Texto
#tokens = nltk.wordpunct_tokenize(raw)
#tokens = tokens[20:1834]
#text = nltk.Text(tokens)
# Normaliza las palabras y construye un vocabulario
#Convierte texto en vocabulario
#words = [w.lower() fow w in text]
#vocab = sorted(set(words))

#Cuando un string tiene más de una línea se tiene que añadir un backlash para que lo cuente como dos líneas o paréntesis
#Con un triple coute vale los saltos de línea sin tener ningun problema
#couplet = """Shall I compare thee to a Sumer's day? 
#Thou are more lovely and more temperate:"""
#print(couplet)
# + para concatenar strings
# * para multiplicar el string
# método find para hayar substring en un string

# CHULETA DE METODOS STRINGS
#s.find(t)    index of first instance of string t inside s (-1 if not found)
#s.rfind(t)    index of last instance of string t inside s (-1 if not found)
#s.index(t)    like s.find(t) except it raises ValueError if not found
#s.rindex(t)    like s.rfind(t) except it raises ValueError if not found
#s.join(text)    combine the words of the text into a string using s as the glue
#s.split(t)    split s into a list wherever a t is found (whitespace by default)
#s.splitlines()    split s into a list of strings, one per line
#s.lower()    a lowercased version of the string s
#s.upper()    an uppercased version of the string s
#s.title()    a titlecased version of the string s
#s.strip()    a copy of s without leading or trailing whitespace
#s.replace(t, u)    replace instances of t with u inside s

#La gran diferencia entre listas y strings es que los primeros son mutables y los segundos no
#Traducir un lenguaje cualquiera a unicode es decoding y a reves es unicode

#La función find encuentra el fichero en el path dado
#path = nltk.data.find('corpora/unicode_samples/polish-lat2.txt')
#f = open(path, encoding='latin2')
#El módulo unicodedata permite inspeccionar propiedades de caracteres en unicode´
#import unicodedata
#lines = open(path, encoding='latin2').readlines()
#line = lines[2]
#print(line.encode('unicode_escape'))
#for c in line: 
#    if ord(c) > 127:
#        print('{} U+{:04x} {}'.format(c.encode('utf8'),ord(c),unicodedata.name(c)))


# 3.4 EXPRESIONES REGULARES PARA DETECTAR PATRONES DE PALABRAS

#Se usa la librería re, y limpiamos los corpus de nombres propios
import re
#wordlist = [w for w in nltk.corpus.words.words('en') if w.islower()]

#Vamosa buscar patrones en un texto, en este caso palabras que acaben con -ed
#re.search(p,s) busca el patron p en el string s, indicamos los caracteres y el dolar indica que es el final de la palabra
#print([w for w in wordlist if re.search('ed$', w)])
#Podemos utilizar . como un comodin que vale como cualquier caracter
#El caracter ^ indica el principio de una palabra
#print([w for w in wordlist if re.search('^..j..t..$',w)])
#El símbolo ? indica que el caracter anterior es opcional de cara a buscar las ocurrencias
#print(sum(1 for w in text if re.search('^e-?mail$',w)))

#Textónimos son dos o más palabras que se introducen con una misma secuencia de teclas
#Para ver que palabras se pueden obtener de la misma manera usamos expresiones regulares
#print([w for w in wordlist if re.search('^[ghi][mno][jlk][def]$',w)])
#Otras búsquedas
#print([w for w in wordlist if re.search('^[ghijklmno]+$',w)])
#El - se usa para indicar un intervalo de valores
#El + se usa para indicar que un conjunto de caracteres se puede usar una o varias veces
#print([w for w in wordlist if re.search('^[g-o]+$',w)])
#print([w for w in wordlist if re.search('^[a-fj-o]+$',w)])
#El + se puede sustituir por * que es 0 o más instancias de este elemento
#+ y * se llaman Kleene closures
#^ tiene una funcion distinta cuando está dentro de la lista, significa que hace match cuando es algo distinto a lo que hay ahí
#print([w for w in wordlist if re.search('^[^aeiouAEIOU]+$',w)])

#wsj = sorted(set(nltk.corpus.treebank.words()))
#Este busca unicamente cifras decimales
#\ ignora el significado especial del siguiente caracter
#print([w for w in wsj if re.search('^[0-9]+\.[0-9]+$', w)])
#Busca palabras seguidas del símbolo del dolar
#print([w for w in wsj if re.search('^[A-Z]+\$$', w)])
#Busca números de 4 cifras
#Los {} especifican el número de repeticiones de los elementos anteriores
#print([w for w in wsj if re.search('^[0-9]{4}$', w)])
#Indica un conjunto de hasta 3 numeros seguidos de un guión y hasta 5 letras
#print([w for w in wsj if re.search('^[0-9]+-[a-z]{3,5}$', w)])
#Tres palabras separadas por guiones y con cardinalidad definida
#print([w for w in wsj if re.search('^[a-z]{5,}-[a-z]{2,3}-[a-z]{,6}$', w)])
#| indica una elección entre los elementos de la derecha y la izquierda
#() indica a que afecta un operador 
#Palabras que terminen en ed o ing
#print([w for w in wsj if re.search('(ed|ing)$', w)])

#CHULETA DE CARACTERES DE EXPRESIONES REGULARES
#.    Wildcard, matches any character
#^abc    Matches some pattern abc at the start of a string
#abc$    Matches some pattern abc at the end of a string
#[abc]    Matches one of a set of characters
#[A-Z0-9]    Matches one of a range of characters
#ed|ing|s    Matches one of the specified strings (disjunction)
#*    Zero or more of previous item, e.g. a*, [a-z]* (also known as Kleene Closure)
#+    One or more of previous item, e.g. a+, [a-z]+
#?    Zero or one of the previous item (i.e. optional), e.g. a?, [a-z]?
#{n}    Exactly n repeats where n is a non-negative integer
#{n,}    At least n repeats
#{,n}    No more than n repeats
#{m,n}    At least m and no more than n repeats
#a(b|c)+    Parentheses that indicate the scope of the operators

#Extrayendo trozos de palabras con ER
#Método findall encuentra todos los matches de las expresiones regulares dadas
#este código busca todas las vocales en una palabra 
#word = 'supercalifragilisticexpialidocious'
#print(re.findall(r'[aeiou]', word))
#print(len(re.findall(r'[aeiou]', word)))
#Busca secuencias de dos o mas vocales en un texto y determina frecuencia relativa
#wsj = sorted(set(nltk.corpus.treebank.words()))
#fd = nltk.FreqDist(vs for word in wsj
 #                     for vs in re.findall(r'[aeiou]{2,}', word))
#fd.most_common(12)

#Al igual que findall sirve para extraer palabras .join() se usa para juntarlas
#regexp = r'^[AEIOUaeiou]+|[AEIOUaeiou]+$|[^AEIOUaeiou]'
#def compress(word):
#    pieces = re.findall(regexp, word)
#    return ''.join(pieces)
#english_udhr = nltk.corpus.udhr.words('English-Latin1')
#print(nltk.tokenwrap(compress(w) for w in english_udhr[:75]))

#Obtenemos diccionario de Rotokas
#rotokas_words = nltk.corpus.toolbox.words('rotokas.dic')
#Bucle que busca en el conjunto de rotokas_words la ER que junta un conjunto de consonantes con las vocales
#cvs = [cv for w in rotokas_words for cv in re.findall(r'[ptksvr][aeiou]', w)]
#Obtenemos la frecuencia condicional distribuida de este conjunto anterior
#cfd = nltk.ConditionalFreqDist(cvs)
#cfd.tabulate()

# Procesa cada palabra w y obtiene cada substring que haga match con la er 
#USando nltk.Index() podemos convertir esto en un índice
#cv_word_pairs = [(cv,w) for w in rotokas_words
#                        for cv in re.findall(r'[ptksvr][aeiou]',w)]
#cv_index = nltk.Index(cv_word_pairs)
#print(cv_index['su'])
#print(cv_index['po'])

# Como obtener la raíz de una palabra
#Este método elimina cualquier cosa que parezca un sufijo 
#def stem(word):
#    for suffix in ['ing','ly','ed','ious','ies','ive','es','s','ment']:
#        if word.endswith(suffix):
#            return word[:-len(suffix)]
#        return word

#Se pueden emplear también expresiones regulares para esta tarea
#print(re.findall(r'^.*(ing|ly|ed|ious|ies|ive|es|s|ment)$','processing'))
#Solo devuelve el sufijo en vez de la palabra completa porque los paréntesis seleccionan substrings
#Si se usa el parentesis para el scope pero no seleccionar material del output se añade ? 
#Ahora si que devuelve la palabra completa
#print(re.findall(r'^.*(?:ing|ly|ed|ious|ies|ive|es|s|ment)$','processing'))
#Ahora vamos a partirlo en raíz y sufijo
#print(re.findall(r'^(.*)(ing|ly|ed|ious|ies|ive|es|s|ment)$','processing'))
#print(re.findall(r'^(.*)(ing|ly|ed|ious|ies|ive|es|s|ment)$','processes'))
#El operador es greedy y .* consume todo lo del input posible
#Esta es una versión no greedy que funciona hasta con sufijo vacío
#print(re.findall(r'^(.*?)(ing|ly|ed|ious|ies|ive|es|s|ment)$', 'processes'))
#print(re.findall(r'^(.*?)(ing|ly|ed|ious|ies|ive|es|s|ment)?$', 'language'))

def stem(word):
    regexp = r'^(.*?)(ing|ly|ed|ious|ies|ive|es|s|ment)?$'
    stem,suffix = re.findall(regexp, word)[0]
    return stem
#raw = """ ENNIS: Listen, strange women lying in ponds distributing swords is no basis for a system of government.  Supreme executive power derives from a mandate from the masses, not from some farcical aquatic ceremony."""
#tokens = word_tokenize(raw)
#print([stem(t) for t in tokens])
# Este método falla algunas veces pero es aceptable

# Se pueden usar er para buscar varias palabras juntas en un texto
#<> se usan para delimitar un token y los espacios en blanco se ignoran
#from nltk.corpus import gutenberg, nps_chat
#moby = nltk.Text(gutenberg.words('melville-moby_dick.txt'))
#Esto buscará las palabras entre a y man
#print(moby.findall(r"<a> (<.*>) <man>"))
#chat = nltk.Text(nps_chat.words())
#Busca Las dos palabras que preceden a bro
#print(chat.findall(r"<.*> <.*> <bro>"))
#Busca las palabras que son de más de tres caracteres y empiezan por l 
#print(chat.findall(r"<l.*>{3,}"))

#Este método devuelve marcadores alrededor de las palabras que cumplen la expresion regular 
#TODO: No está marcando nada y unicamente me imprime el texto 
#regexp = r"<.*> <.*> <bro>"
#texto = "jolin recorcholis bro es increible esto que me cuentas bro"
#nltk.re_show(regexp,texto)
#Interfaz gráfica que explora expresiones regulares
#nltk.app.nemo()

#Programa de descubrimiento de hiperónimos
#Produce algunos falsos negativos
#from nltk.corpus import brown
#hobbies_learned = nltk.Text(brown.words(categories=['hobbies','learned']))
#print(hobbies_learned.findall(r"<\w*> <and> <other> <\w*s>"))

#Normalización de texto
#La tarea de poner todo un texto en minúscula es la normalizacion
#Eliminar los afijos se conoce como stemming 
#Lemmatización es la tarea de asegurarse de que una palabra es de un diccionario 

#Uso de varios stemmers de nltk
#porter = nltk.PorterStemmer()
#lancaster = nltk.LancasterStemmer()
#print([porter.stem(t) for t in tokens])
#print([lancaster.stem(t) for t in tokens])
# Se elije un stemmer que convenga para la aplicación con la que se está trabajando
#Porter es bueno para indexar textos y soportar texto y formas alternativas de palabras

 #Indexando un texto usando un Stemmer   
class IndexedText(object):

    def __init__(self, stemmer, text):
        self._text = text
        self._stemmer = stemmer
        self._index = nltk.Index((self._stem(word), i)
                                 for (i, word) in enumerate(text))

    def concordance(self, word, width=40):
        key = self._stem(word)
        wc = int(width/4)                # words of context
        for i in self._index[key]:
            lcontext = ' '.join(self._text[i-wc:i])
            rcontext = ' '.join(self._text[i:i+wc])
            ldisplay = '{:>{width}}'.format(lcontext[-width:], width=width)
            rdisplay = '{:{width}}'.format(rcontext[:width], width=width)
            print(ldisplay, rdisplay)

    def _stem(self, word):
        return self._stemmer.stem(word).lower()

#porter = nltk.PorterStemmer()
#grail = nltk.corpus.webtext.words('grail.txt')
#text = IndexedText(porter,grail)
#text.concordance('lie')

#Lemmatización 
#wnl = nltk.WordNetLemmatizer()
#print([wnl.lemmatize(t) for t in tokens])
#Otra tarea de normalización es identificar palabras no estandar como numeros, abreviaciones y fechas, que se deben de asignar a un vocabulario dustinto para mantenerlo pequeño

#Expresiones regulares para la tokenización de texto
#raw = """'When I'M a Duchess,' she said to herself, (not in a very hopeful tone
#though), 'I won't have any pepper in my kitchen AT ALL. Soup does very
#well without--Maybe it's always pepper that makes people hot-tempered,'..."""
#Hace match con todo al ser un espacio en blanco lo que contiene
#print(re.split(r' ', raw))
#Hace match con un espacio,tabulador(\t) o salto de línea (\n)  
#print(re.split(r'[ \t\n]+', raw))
#\s+ representa cualquier caracter de espacio
#print(re.split(r'\s+', raw))    
#Se usala notación \w+ para referirnos a combinaciones alfanumericas y
#\W+ para notacion que es lo contrario a combinacion alfanumerico
#print(re.split(r'\W+', raw))
#Esta busca match con una secuencia de palabras, si no cuela busca in non-whitespace
#\S es el opuesto de \s seguido de más palabras
#print(re.findall(r'\w+|\S\w*',raw))
#Esta expresion permite palabras, guiones internos y apóstrofes 
#print(re.findall(r"\w+(?:[-']\w+)*|'|[-.(]+|\S\w*", raw))

#CHULETA DE SIMBOLOS DE EXPRESIONES REGULARES
#\b    Word boundary (zero width)
#\d    Any decimal digit (equivalent to [0-9])
#\D    Any non-digit character (equivalent to [^0-9])
#\s    Any whitespace character (equivalent to [ \t\n\r\f\v])
#\S    Any non-whitespace character (equivalent to [^ \t\n\r\f\v])
#\w    Any alphanumeric character (equivalent to [a-zA-Z0-9_])
#\W    Any non-alphanumeric character (equivalent to [^a-zA-Z0-9_])
#\t    The tab character
#\n    The newline character

#nltk.regexp_tokenize() es más eficiente, rompemos las expresiones regulares en varias líneas
#?x le dice a python que quite los espacios y comentarios
#text = 'That U.S.A poster-print costs $12.40...'
#pattern = r'''(?x)         #set flag to allow verbose regexps
#    (?:[A-Z]\.)+           #abbreviations, e.g. U.S.A
#    | \w+(?:-\w+)*         #words with optional internal hyphens
#    | \$?\d+(?:\.\d+)?%?   #currency and percentages, e.g. $12.40, 82%
#    | \.\.\.               #ellipsis
#    | [][.,;"'?():-_`]     #these are separate tokens; includes ], [
#    '''
#print(nltk.regexp_tokenize(text,pattern))    
#Al usar ?x no se puede volver a usar el ' ' para un caracter de espacio y se usa \s     
#Podemos evaluar un tokenizador comparando sus resultados con una lista de palabras´

#3.8 SEGMENTATION
#Para dividir la oracion en tokens primero se debe de dividir en oraciones
len(nltk.corpus.brown.words()) / len(nltk.corpus.brown.sents())
#Para dividir en oraciones usamos Punky de NLTK
#Pretty print sirve para imprimir bien
text = nltk.corpus.gutenberg.raw('chesterton-thursday.txt')
sents = nltk.sent_tokenize(text)
pprint.pprint(sents[79:89])
#Segmentacion de palabras
#Cuando en procesamiento del lenguaje oral se segmentan todas las palabras juntas 
#Hay que separar el contenido del texto de la segmentación, se puede hacer con un valor booleano para indicar si hay un cambio de palabra después del caracter
#Se puede tener otro conjunto de booleanos además que marque el fin de oraciones 
text = "doyouseethekittyseethedoggydoyoulikethekittylikethedoggy"
seg1 = "0000000000000001000000000010000000000000000100000000000"
seg2 = "0100100100100001001001000010100100010010000100010010000"
#Metodo que segmenta el texto en oraciones o palabras
def segment(text, segs):
    words = []
    last = 0
    for i in range(len(segs)):
        if segs[i] == '1':
            words.append(text[last:i+1])
            last = i+1
    words.append(text[last:])
    return words

print(segment(text, seg1))
print(segment(text, seg2))
#UNa funcion objetiva proporciona una función de puntuacion cuyo valor se intenta optimizar
#Esta basado en el tamaño del lexicon y la informacion necesaria para reconstruir el texto fuente.
#Implementación de esto
def evaluate(text, segs):
    words = segment(text, segs)
    text_size = len(words)
    lexicon_size = sum(len(word) + 1 for word in set(words))
    return text_size + lexicon_size
text = "doyouseethekittyseethedoggydoyoulikethekittylikethedoggy"
seg1 = "0000000000000001000000000010000000000000000100000000000"
seg2 = "0100100100100001001001000010100100010010000100010010000"
seg3 = "0000100100000011001000000110000100010000001100010000001"
print(segment(text, seg3))
print(evaluate(text, seg3))
print(evaluate(text,seg2))
print(evaluate(text, seg1))

#Programa que busca el patron de 0s y 1s para minimizar la función objetivo
from random import randint

def flip(segs, pos):
    return segs[:pos] + str(1-int(segs[pos])) + segs[pos+1:]

def flip_n(segs, n):
    for i in range(n):
        segs = flip(segs, randint(0, len(segs)-1))
    return segs

def anneal(text, segs, iterations, cooling_rate):
    temperature = float(len(segs))
    while temperature > 0.5:
        best_segs, best = segs, evaluate(text, segs)
        for i in range(iterations):
            guess = flip_n(segs, round(temperature))
            score = evaluate(text, guess)
            if score < best:
                best, best_segs = score, guess
        score, segs = best, best_segs
        temperature = temperature / cooling_rate
        print(evaluate(text, segs), segment(text, segs))
    print()
    return segs

print(anneal(text, seg1, 5000, 1.2))

#3.9 Formateo, de Listas a Strings
#Para convertir una lista de strings en un solo string usamos el método join
#Determina lo que esta entre comillas que ponemos entre las diferentes palabras del string
silly = ['We', 'called', 'him', 'Tortoise', 'because', 'he', 'taught', 'us', '.']
''.join(silly)
';'.join(silly)
''.join(silly)
#Escribimos los resultados en un fichero
output_file = open('output.txt','w')
words = set(nltk.corpus.genesis.words('english-kjv.txt'))
for word in sorted(words):
    print(word, file=output_file)
    



