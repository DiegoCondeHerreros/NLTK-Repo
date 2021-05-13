# -*- coding: utf-8 -*-
'''
Created on 2 mar. 2021

@author: DIEGO
'''
import nltk
#1. USANDO UN TAGGER
#Un Part of Speech Tagger o POS procesa una secuencia de palabras y le añade un tag
from nltk.tokenize import word_tokenize
from _operator import itemgetter
from nltk.app.wordfreq_app import plot_word_freq_dist
#text = word_tokenize("And now for something completely different")
#print(nltk.pos_tag(text))
#and -> CC -> Conjuncion de coordinación
#now -> RB -> Adverbios
#for -> IN -> Preposición
#something -> NN -> Nombre
#different -> JJ -> Adjetivo

#Se usa el siguiente comando para informarse sobre la tag
#print(nltk.help.upenn_tagset('RB'))
#text = word_tokenize("They refuse to permit us to obtain the refuse permit")
#print(nltk.pos_tag(text))
#Las palabras pueden aparecer como diferentes tipos de palabras ya que son homófonas
#Similar te muestra las palabras que aparecen en un texto en el mismo contexto que la dada 
#text = nltk.Text(word.lower() for word in nltk.corpus.brown.words())
#print(text.similar('woman'))
#print(text.similar('bought'))
#print(text.similar('over'))
#print(text.similar('the'))


#2 TAGGED CORPORA


#2.1 Representando tokens etiquetados
#Los tokens taggeados están formados por una tupla que es el token y la etiqueta
#Con la libreria str2tuple se puede crear una instancia de esto
#tagged_token = nltk.tag.str2tuple('fly/NN')
sent = '''
    The/AT grand/JJ jury/NN commented/VBD on/IN a/AT number/NN of/IN
    other/AP topics/NNS ,/, AMONG/IN them/PPO the/AT Atlanta/NP and/CC
    Fulton/NP-tl County/NN-tl purchasing/VBG departments/NNS which/WDT it/PPS
    said/VBD ``/`` ARE/BER well/QL operated/VBN and/CC follow/VB generally/RB
    accepted/VBN practices/NNS which/WDT inure/VB to/IN the/AT best/JJT
    interest/NN of/IN both/ABX governments/NNS ''/'' ./.
    '''
#print([nltk.tag.str2tuple(t) for t in sent.split()])


#2.2 Leyendo Corpora Etiquetados
#Hay corpus que vienen etiquetados de base
#print(nltk.corpus.brown.tagged_words())
#print(nltk.corpus.brown.tagged_words(tagset='universal'))
#print(nltk.corpus.nps_chat.tagged_words())
#print(nltk.corpus.conll2000.tagged_words())
#print(nltk.corpus.treebank.tagged_words())
#print(nltk.corpus.treebank.tagged_words(tagset='universal'))
#Incluye varios idiomas, python de base lo imprime en hexadecimal
 
#2.3 Universal Part-of-Speech Tagset 

#ADJ    adjective    new, good, high, special, big, local
#ADP    adposition    on, of, at, with, by, into, under
#ADV    adverb    really, already, still, early, now
#CONJ    conjunction    and, or, but, if, while, although
#DET    determiner, article    the, a, some, most, every, no, which
#NOUN    noun    year, home, costs, time, Africa
#NUM    numeral    twenty-four, fourth, 1991, 14:24
#PRT    particle    at, on, out, over per, that, up, with
#PRON    pronoun    he, their, her, its, my, I, us
#VERB    verb    is, say, told, given, playing, would
#.    punctuation marks    . , ; !
#X    other    ersatz, esprit, dunno, gr8, univeristy

from nltk.corpus import brown
brown_news_tagged = brown.tagged_words(categories='news',tagset='universal')
tag_fd = nltk.FreqDist(tag for (word,tag) in brown_news_tagged)
#print(tag_fd.most_common())
#tag_fd.plot(cumulative=True)
#Se puede hacer busquedas con estas tags usando el método nltk.app.concordance() 

#2.4 Nombres
#Tags simplificados son N para nombres comunes y NP para nombres propios
#Programa que indica que tipos de palabra suelen ser anteriores al nombre
word_tag_pairs = nltk.bigrams(brown_news_tagged)
noun_preceders = [a[1] for (a,b) in word_tag_pairs if b[1]=='NOUN']
fdist = nltk.FreqDist(noun_preceders)
#print([tag for (tag,_) in fdist.most_common()])

#2.5 Verbos
#Verbos que más aparecen en un texto
wsj = nltk.corpus.treebank.tagged_words(tagset='universal')
word_tag_fd=nltk.FreqDist(wsj)
#print([wt[0] for (wt,_) in word_tag_fd.most_common() if wt[1]=='VERB'])
# Como se está accediendo a una tupla de elementos se da que se accede a los difrentes significados de una palabra
cfd1 = nltk.ConditionalFreqDist(wsj)
#print(cfd1['yield'].most_common())
#print(cfd1['cut'].most_common())
#Se puede alterar de tal modo que la tag sea la condicion
#Programa que mira las palabras probables para una tag dada
wsj = nltk.corpus.treebank.tagged_words()
cfd2 = nltk.ConditionalFreqDist((tag,word) for (word,tag) in wsj)
#print(list(cfd2['VBN']))
#La tag VBD es pasado simple y VBN es el pasado participio
#Programa que muestra qye palabras son  a la vez VBN y VBD
#print([w for w in cfd1.conditions() if 'VBD' in cfd1[w] and 'VBN' in cfd1[w]])
idx1 = wsj.index(('kicked','VBD'))
#print(wsj[idx1-4:idx1+1])
idx2 = wsj.index(('kicked','VBN'))
#print(wsj[idx2-4:idx2+1])

#2.6 Adjetivos y Adverbios
#2.7 Unimplified Tags
def findtags(tag_prefix,tagged_text):
    cfd = nltk.ConditionalFreqDist((tag,word) for (word,tag) in tagged_text
                                   if tag.startswith(tag_prefix))
    return dict((tag,cfd[tag].most_common(5)) for tag in cfd.conditions())

#tagdict = findtags('NN',nltk.corpus.brown.tagged_words(categories='news'))
#for tag in sorted(tagdict):
#    print(tag,tagdict)

#Muchas variantes de los NN 
# $ se usa para nombres posesivos
# S se usa para nombres plurales
# P para nombres propios
#Las tags también tienen modificadores de sufijos
# -NC para citas
# -HL para palabras en cabeceras
# -TL para los títulos

#2.8 Explorando Corpora etiquetados
brown_learned_text = brown.words(categories='learned')
#Para ver el contexto de often accedemos a las palabras que suelen ir después de often
#print(sorted(set(b for (a,b) in nltk.bigrams(brown_learned_text) if a =='often')))
#Usamos el método tagged_words() para comprobar la tag de estas palabras 
brown_lrnd_tagged = brown.tagged_words(categories='learned', tagset='universal')
tags = [b[1] for (a,b) in nltk.bigrams(brown_lrnd_tagged) if a[0] == 'often']
fd = nltk.FreqDist(tags)
#fd.tabulate()

#Con este programa buscamos trios de palabras que sean verbo + to `verbo
def process(sentence):
    for(w1,t1), (w2,t2), (w3,t3) in nltk.trigrams(sentence):
        if(t1.startswith('V') and t2 == 'TO' and t3.startswith('V')):
            print(w1,w2,w3)
#for tagged_sent in brown.tagged_sents():
#    process(tagged_sent)
        
#Programa que busca las palabras más ambiguas del texto
#data = nltk.ConditionalFreqDist((word.lower(),tag)
#                                for (word,tag) in brown_news_tagged)
#for word in sorted(data.conditions()):
#    if len(data[word]) > 3:
#        tags = [tag for (tag,_) in data[word].most_common()]
#        print(word,''.join(tags))
#Esta app permite ver el contexto de las palabras en un corpus
#nltk.app.concordance()        


#3 Mapear palabras a propiedades usando diccionarios de python
#3.1 Listas indexadas vs Diccionarios
pos = {}
pos['colorless'] = 'ADJ'
pos['ideas'] = 'N'
pos['sleep'] = 'V'
pos['furiously'] = 'ADV'
#print(pos)
#print(pos['green'])
list(pos)
sorted(pos)
#print([w for w in pos if w.endswith('s')])
#for word in sorted(pos):
#    print(word + ";",pos[word])
#keys(), values() y items() permiten acceder a keys, values y key-value respectivamente
#3.3 Maneras de definir un diccionario
pos = {'colorless':'ADJ','ideas':'N','sleep':'V','furiously':'ADV'}
pos = dict(colorless='ADJ', ideas='N',sleep='V',furiously='ADV')
#Las keys tienen que ser tipos de datos inmutables
#Un default dictionary es aquel que cuando le das una clave que no existe te la crea
from collections import  defaultdict
frecuency = defaultdict(int)
frecuency['colorless']=4
#print(frecuency['ideas'])
pos=defaultdict(list)
pos['sleep']=['NOUN','VERB']
#print(pos['ideas'])
pos = defaultdict(lambda:'NOUN')
pos['colorless']='ADJ'
#print(pos['blog'])
#print(list(pos.items()))

#Programa que crea un diccionario default que mapea cada palabra con su sustituta 
#Las n palabras más frecuentes se mapearan a si mismas
alice = nltk.corpus.gutenberg.words('carroll-alice.txt')
vocab = nltk.FreqDist(alice)
v1000=[word for (word,_) in vocab.most_common(1000)]
mapping=defaultdict(lambda:'UNK')
for v in v1000:
    mapping[v]=v
alice2=[mapping[v] for v in alice]
#print(alice2[:100])

#3.5 Actualizar un diccionario de forma incremental
#Inicializamos diccionario vacío y luego se procesa un texto y si una tag no se ha visto antes se actualiza el diccionario
counts = defaultdict(int)
for (word,tag) in brown.tagged_words(categories='news',tagset='universal'):
    counts[tag]+=1
#print(counts['NOUN'])
#print(sorted(counts))
from operator import itemgetter
#print(sorted(counts.items(),key=itemgetter(1),reverse=True))
#print([t for t, c in sorted(counts.items(),key=itemgetter(1),reverse=True)])

pair = ('NP',8336)
#print(pair[1])
#print(itemgetter(1)(pair))

#3.6 Claves y Valores Complejos
#El valor por defecto de la entrada es otro diccionario
pos = defaultdict(lambda: defaultdict(int))
#Se itera sobre las parejas de palabras-tags
for((w1,t1),(w2,t2)) in nltk.bigrams(brown_news_tagged):
    #Actualizamos los vlaores
    pos[(t1,w2)][t2] += 1
#Se accede al objeto tipo diccionario
#print(pos[('DET','right')])


#3.7 Invirtiendo el diccionario
#Obtenemos una clave partiendo de n valor
counts = defaultdict(int)
for word in nltk.corpus.gutenberg.words('milton-paradise.txt'):
    counts[word]+=1
#print([key for (key,value) in counts.items() if value==32])

pos = {'colorless':'ADJ','ideas':'N','sleep':'V','furiously':'ADV'}
pos2=dict((value,key) for (key,value) in pos.items())
#print(pos2['N'])

#Esta técnica no funcionará con update y habrá que usar append
pos.update({'cats':'N','scratch':'V','peacefully':'ADV','old':'ADJ'})
pos2=defaultdict(list)
for key,value in pos.items():
    pos2[value].append(key)
#print(pos2['ADV'])
#NLTK tiene metodos para este tipo de indexado
pos2=nltk.Index((value,key) for (key,value) in pos.items())
#print(pos2['ADV'])

#CHULETA COMANDOS DICCIONARIOS PYTHON
#d = {}    create an empty dictionary and assign it to d
#d[key] = value    assign a value to a given dictionary key
#d.keys()    the list of keys of the dictionary
#list(d)    the list of keys of the dictionary
#sorted(d)    the keys of the dictionary, sorted
#key in d    test whether a particular key is in the dictionary
#for key in d    iterate over the keys of the dictionary
#d.values()    the list of values in the dictionary
#dict([(k1,v1), (k2,v2), ...])    create a dictionary from a list of key-value pairs
#d1.update(d2)    add all items from d2 to d1
#defaultdict(int)    a dictionary whose default value is zero


#4. Etiquetado automático
#Como se tiene que comprobar el rol de una palabra en un texto se trabajará a nivel de oración 
#brown_tagged_sents = brown.tagged_sents(categories='news')
#brown_sents=brown.sents(categories='news')

#4.1 Etiquetador por defecto
#El clasificador más sencillo es aquel que asigna a todo la misma tag
#A partir de aquí lo mejoramos de tal modo que asigne la tag más probable
#Cual es la tag más probable de todas
#tags = [tag for (word,tag) in brown.tagged_words(categories='news')]
#print(nltk.FreqDist(tags).max())
#Creamos etiquetador que ponga todo como un nombre
#raw = "I do not like green eggs and ham, I do not like them Sam I am!"
#tokens= nltk.word_tokenize(raw)
#default_tagger=nltk.DefaultTagger('NN')
#print(default_tagger.tag(tokens))
#Con este método comprobamos la eficacia del clasificador
#print(default_tagger.evaluate(brown_tagged_sents))

#4.2 El Etiquetador de expresiones regulares
#Los utilizamos para clasificar según patrones linguisticos
patterns = [
    (r'.*ing$','VGB'), #Gerundios
    (r'.*ed$','VBD'),  #Pasado simple
    (r'.*es$','VBZ'),   #Presente de 3a persona del singular
    (r'.*ould$','MD'),  #Modales
    (r'.*\'s$','NN$'),  #Nombres posesivos
    (r'.*s$','NNS'),    #Nombres plurales
    (r'^-?[0-9]+(\.[0-9]+)?$','CD'), #Numeros cardinales
    (r'.*','NN')        #Nombres a secas
    ]
#regxp_tagger = nltk.RegexpTagger(patterns)
#print(regxp_tagger.tag(brown_sents[3]))
#print(regxp_tagger.evaluate(brown_tagged_sents))
#Ahora tiene mejores resultados al haber definido tipos de verbo y nombres
#La ultima ER lo que hace es pillar todo lo que no encaje como nombre
#Una manera de mejorar esto sería incluir ER para adjetivos y adverbios

#4.3 El Etiquetador Lookup
#Vamos a buscar las 100 palabras más frecuentes y guardar su más probable etiqueta
#fd = nltk.FreqDist(brown.words(categories='news'))
#cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
#most_freq_words=fd.most_common(100)
#likely_tags=dict((word,cfd[word].max()) for (word,_) in most_freq_words)
#baseline_tagger = nltk.UnigramTagger(model=likely_tags)
#print(baseline_tagger.evaluate(brown_tagged_sents))
#Mejora mucho los resultados, clasifica casi la mitad bien
#El problema es que a los que no son de las 100 palabras más usadas no se asigna NN
#Lo suyo es usar primero la tabla de lookup y luego un etiquetador por defecto, este proceso se conoce como backoff
#se hace especificando un etiquetador como parámetro del otro
#baseline_tagger = nltk.UnigramTagger(model=likely_tags,backoff=nltk.DefaultTagger('NN'))
#Programa que evalua los etiquetadores de lokup con un rango de tamaños
def performance(cfd,wordlist):
    lt = dict((word,cfd[word].max())for word in wordlist)
    baseline_tagger=nltk.UnigramTagger(model=lt,backoff=nltk.DefaultTagger('NN'))
    return baseline_tagger.evaluate(brown.tagged_sents(categories='news'))

def display():
    import pylab
    word_freqs=nltk.FreqDist(brown.words(categories='news')).most_common()
    words_by_freq = [w for (w,_) in word_freqs]
    cfd=nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
    sizes=2**pylab.arange(15)
    perfs=[performance(cfd,words_by_freq[:size])for size in sizes]
    pylab.plot(sizes,perfs,'-bo')
    pylab.title('Lookup Tagger Performance with Varying Model Size')
    pylab.xlabel('Model Size')
    pylab.ylabel('Performance')
    pylab.show()

#display()    

#5.Etiquetado de N-gramas
#5.1 Unigram Tagging 
#Este asigna a una palabra la categoria a la que es más probable que pertenezca
#Es como un lookup tagger excepto porque es mejor para entrenar
brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents=brown.sents(categories='news')
unigram_tagger=nltk.UnigramTagger(brown_tagged_sents)
#print(unigram_tagger.tag(brown_sents[2007]))
#print(unigram_tagger.evaluate(brown_tagged_sents))
#Tiene un 93% de aciertos
#Se entrena un UnigramTagger pasando frases etiquetadas como parámetro cuando se inicializa el etiquetador
#Luego inspecciona la etiqueta de cada palabra y guardando la etiqueta más probable de una palabra en un diccionario que se guarda en el objeto del etiquetador

#5.2 Separando los datos de entrenamiento y los de prueba
#Se debe de separar porque si no no hay ningún logro en memorizar datos
#En este ejemplo se separa en un 90% de entrenamiento y un 10% de prueba
size=int(len(brown_tagged_sents)*0.9)
#print(size)
train_sents=brown_tagged_sents[:size]
test_sents=brown_tagged_sents[size:]
unigram_tagger=nltk.UnigramTagger(train_sents)
#print(unigram_tagger.evaluate(test_sents))
#La puntuación es peor pero nos da una idea mejor del etiquetador

#5.3 Etiquetado general de N-Gramas
#En unigramas se usa unicamente un elemento del contexto siendo este el token actual
#Con esto solo se puede saber que es una etiqueta a priori
#Con N-gramas se tiene en cuenta no solo la palabra sino las n-1 tags que la preceden
#1-gram=unigrama,2-gram=bigrama,3-gram=trigrama etc etc
#Ejemplo con bigrama y entrenamiento previo
bigram_tagger=nltk.BigramTagger(train_sents)
#print(bigram_tagger.tag(brown_sents[2007]))
unseen_sent=brown_sents[4203]
#print(bigram_tagger.tag(unseen_sent))
#Aquí el contexto es triki porque si se encuentra una palabra que ya ha visto antes con una palabra anterior que no reconoce la etiqueta como none
#print(bigram_tagger.evaluate(test_sents))
#A medida que se aumenta el valor de N lo específico del contexto aumenta a su vez y le cuesta más etiquetar
#A esto se le llama el problema de los datos dispersos a medida que se aumenta la presicion disminuye los resultados que se cubren.

#5.4 Combinación de etiquetadores
#La eficiencia perfecta suele ser cuando se combinan los más eficientes con los más amplios
#1. Intentamos etiquetar los tokens con el etiquetador de bigramas
#2. Si el bigrama falla buscando la etiqueta se usa el de unigramas
#3. Si este falla a su vez se usa el etiquetador por defecto
#Se indica esto en el código con el parámetro backoff
t0=nltk.DefaultTagger('NN')
t1=nltk.UnigramTagger(train_sents,backoff=t0)
t2=nltk.BigramTagger(train_sents,backoff=t1)
t3=nltk.TrigramTagger(train_sents,backoff=t2)
#print(t2.evaluate(test_sents))
#print(t3.evaluate(test_sents))
#nltk.BigramTagger(sents,cutoff=2,backoff=t1) nos descarta los contextos que se hayan visto unicamente una o dos veces
#El etiquetador debe ser lo más pequeño posible y descarta cuando los resultados del backoff son los mismos

#5.5 Etiquetando palabras desconocidas
#Según las soluciones hasta ahora le asignaría la etiqueta por defecto, en este caso un nombre
#Otra solución a esto es limitar el contexto del etiquetador a las n más frecuentes y asignar la etiqueta desconocido(UNK) a la que no.
  
#5.6 Guardar etiquetadores
#Para no entrenarlos en cada ejecución puedes guardarlos en un fichero
from pickle import dump
output=open('t2.pk1','wb')
dump(t2,output,-1)
output.close()
#Cargamos nuestro proceso en otro etiquetador guardado
#from pickle import load
#input=open('t2.pk1','rb')
#tagger=load(input)
#input.close()
#Usamos el etiquetador cargado
#texts='''The board's action shows that free enterprise is up against in our complex maze of regulatory laws.'''
#tokens=text.split()
#tagger.tag(tokens)

#5.7 Limites de rendimiento
#Con este código obtenemos los contextos ambiguos
cfd=nltk.ConditionalFreqDist(
    ((x[1],y[1],z[0]),z[1])
    for sent in brown_tagged_sents
    for x,y,z in nltk.trigrams(sent))
ambiguous_contexts=[c for c in cfd.conditions() if len(cfd[c])>1]
#print(sum(cfd[c].N() for c in ambiguous_contexts)/cfd.N())
#Calculamos la matriz de confusion que contiene los errores cometidos
test_tags = [tag for sent in brown.sents(categories='editorial')
                 for (word,tag) in t2.tag(sent)]
gold_tags = [tag for (word,tag) in brown.tagged_words(categories='editorial')]
#print(nltk.ConfusionMatrix(gold_tags,test_tags))


#6. Etiquetado basado en la transformación
#Etiquetado Brill consiste en adivinar la etiqueta de cada palabra y luego volvemos a arreglar los errores
#Por eso se dice que es transformativo, es un método de aprendizaje supervisado ya que se necesita tener los datos de entrenamiento anotados para ver si se cometen errores  
#Por ejemplo  se aplican reglas como reemplazar un NN por VB si la palabra previa es TO
#Otra regla es reemplazar TO con IN cuando la siguiente tag es NNS
#from nltk.tbl import demo as brill_demo
#brill_demo.demo()
#print(open("errors.out").read())


#7. Como determinar la categoria de una palabra
#A nivel morfológico -ness comvierte adjetivo a nombre, -ment comvierte verbo en nombre
#A nivel sintáctico con que palabras van juntas indica que tipo de palabra es
#A nivel semántico






