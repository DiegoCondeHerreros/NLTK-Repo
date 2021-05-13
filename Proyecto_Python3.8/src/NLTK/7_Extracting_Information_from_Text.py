# -*- coding: utf-8 -*-
'''
Created on 9 mar. 2021

@author: DIEGO
'''
import chunk
from nltk.tag.senna import SennaChunkTagger
from test.test_logging import BORING
"""
1 Extracción de información
La información viene a veces en datos estructurados siendo una organizaciín predecebile de entidades y relaciones
En este capítulo se decide de antemano que buscaremos información específica  en el texto
Para esto convertimos información desestructurada en información estrucurada 
La manera de obtener significado de este texto es extracción de información
"""
"""
1.1 Arquitectura de extracción de información
Se comienza procesando un documento usando procedimientos de procesamiento de texto en bruto y categorización y etiquetado de palabras
Segmentamos el texto en oraciones y cada una de estas la segmentamos en palabras con un tokenizador, luego se etiqueta y despues usamos la detección de entidades nombradas
Después en la detección de entidades nombradas buscamos menciones de entidades interesantes en cada oración y finalmente usamos deteccion de relaciones para buscar los elementos comunes entre distintas entidades en el texto.
"""
import nltk,re,pprint
def ie_preprocess(document):
    sentences=nltk.sent_tokenize(document)
    sentences=[nltk.word_tokenize(sent) for sent in sentences]
    sentences=[nltk.pos_tag(sent) for sent in sentences]

"""
2 Chunking
Esta técnica consiste en segmentar y etiquetar secuencias de múltiples tokens
Se organizan en cajas, las más pequeñas son la tokenización a nivel de palabra y etiquetado de part-of-speech mientras uqe las cajas más grandes son de alto nivel.  
Las cajas grandes son chunks, estos suelen seleccionar un subconjunto de tokens, las piezas de un chunker no se sobreponen
En el ejemplo de la imagen chunking es similar a un análisis sintáctico  
"""
"""
2.1 Chunking de sintagmas nominales(NP-chunking)
Ejemplo:
The/DT market/NN ] for/IN [ system-management/NN software/NN ] for/IN [ Digital/NNP ] [ 's/POS hardware/NN ] is/VBZ fragmented/JJ enough/RB that/IN [ a/DT giant/NN ] such/JJ as/IN [ Computer/NNP Associates/NNPS ] should/MD do/VB well/RB there/RB ./.
Las etiquetas son muy útiles para esto 
Para crear un NP-chunker primero definimos un chunk grammar que es un conjunto de reglas que indica como se tienen que chunkear las oracicones
Esto puede hacerse con expresiones regulares.
En el siguiente ejemplo ponemos que un NP chunk se forma cuando encuentra un determinante opcional(DT) seguido de un número de adjetivos(JJ) y finalmente un nombre(NN). Con esta gramática creamos un parser de chunks, y se prueba sobre ejemplos, el resultado es un arbol que se puede imprimir o mostrar graficamente

sentence=[("the","DT"),("little","JJ"),("yellow","JJ"),("dog","NN"),("barked","VBD"),("at","IN"),("the","DT"),("cat","NN")]
grammar="NP: {<DT>?<JJ>*<NN>}"
cp = nltk.RegexpParser(grammar)
result=cp.parse(sentence)
print(result)
result.draw()
"""
"""
2.2 Los Patrones
las reglas de la gramática usan patrones de etiquetas para describir las secuencias de palabras etiquetadas. 
Estos son similares a las expresiones regulares.
El ejemplo anterior puede usar <DT>?<JJ.*>*<NN.*>+.
Este pilla un determinante que es opcional, cualquier numero de adjetivos y uno o más nombres 
Este patrón sigue sin cubrirlo todo y puede ser mejor
Se pueden probar patrones con nltk.app.chunkparser()
"""
"""
2.3 Chunking con expresiones regulares
RegexParser empieza con una estructura plana en la que no hay tokens, las reglas se aplican actualizando la estructura del chunck, cuando se han aplicado todas se devuelve el chunk resultante.
"""
grammar=r"""
    NP: {<DT|PP\$>?<JJ>*<NN>}    #chunk de determinantes/posesivos, adjetivos y nombre
        {<NNP>+}                 #chunk de secuencias de nombres propios   
"""
cp=nltk.RegexpParser(grammar)
sentence=[("Rapunzel", "NNP"), ("let", "VBD"), ("down", "RP"),
                 ("her", "PP$"), ("long", "JJ"), ("golden", "JJ"), ("hair", "NN")]
#print(cp.parse(sentence))
#El $ se usa como caracter especial en expresiones regulares, se tiene que hacer backlash antes de eso
#La izquierda tiene prioridad sobre lo otro 
nouns=[("money", "NN"), ("market", "NN"), ("fund", "NN")]
grammar ="NP: {<NN><NN>} #Chunk dos nombres consecutivos"
cp=nltk.RegexpParser(grammar)
#print(cp.parse(nouns))

"""
2.4 Explorando corporas de texto

cp=nltk.RegexpParser('CHUNK: {<V.*> <TO> <V.*>}')
brown=nltk.corpus.brown
for sent in brown.tagged_sents():
    tree=cp.parse(sent)
    for subtree in tree.subtrees():
        if subtree.label()=='CHUNK': print(subtree)
#Método general que sirve para esto mismo        
from nltk.corpus import brown
def find_chunks(chunk):
    cp=nltk.RegexpParser(chunk)
    for sent in brown.tagged_sents():
        tree=cp.parse(sent)
        for subtree in tree.subtrees():
            if subtree.label()=='CHUNK': print(subtree)
"""
"""
2.5 Chinking
Es más facil definir que es lo que se quiere excluir de un chunk, un chink es el conjunto de tokens que no se incluyen en el chunk
Ejemplo:
[ the/DT little/JJ yellow/JJ dog/NN ] barked/VBD at/IN [ the/DT cat/NN ]
Es el proceso de eliminar tokens de un chunk 
Si abarca todo un chunk esto lo elimina, si está en el medio se eliminan los tokens y lo parte en dos chunks
"""
grammar = r"""
  NP:
    {<.*>+}          # Chunk everything
    }<VBD|IN>+{      # Chink sequences of VBD and IN
  """
sentence = [("the", "DT"), ("little", "JJ"), ("yellow", "JJ"),
       ("dog", "NN"), ("barked", "VBD"), ("at", "IN"),  ("the", "DT"), ("cat", "NN")]
cp = nltk.RegexpParser(grammar)
#print(cp.parse(sentence))
"""
2.6 Representación Chunks: Etiquetas vs Árboles
La representación más común es IOB tags, esto etiqueta con etiquetas especiales Inside, Outside, Begin.
Token marcado con B marca el principio de un chunk, los siguientes se marcan con la I, y los demás con la O.
El B e I tienen sufijos tipo B-NP, I-NP. Imprime una etiqueta y token por línea
También se puede poner en forma de árbol
NLTK no usa los árboles pero tiene métodos para leer y escribir.    
"""

"""
3 Desarrollando y evaluando chunkers
Primero veremos como convertir el formato IOB a un arbol NLTK y luego se hará en una escala mayor con corpus chunkeados.
"""
"""
3.1 Leyendo el formato IOB y el Corpus CoNLL 2000
Las categorias de chunk de este corpus NP, VP, PP.
Una función de conversión chunk.conlstr2tree() construye un arbol viniendo de esta representación 
Nos permite escoger un subconjunto de los tipos del árbol de chunks  

text='''
    he PRP B-NP
    accepted VBD B-VP
    the DT B-NP
    position NN I-NP
    of IN B-PP
    vice NN B-NP
    chairman NN I-NP
    of IN B-PP
    Carlyle NNP B-NP
    Group NNP I-NP
    , , O
    a DT B-NP
    merchant NN I-NP
    banking NN I-NP
    concern NN I-NP
    . . O
    '''
nltk.chunk.conllstr2tree(text, chunk_types=['NP']).draw()
from nltk.corpus import conll2000
print(conll2000.chunked_sents('train.txt')[99])
print(conll2000.chunked_sents('train.txt', chunk_types=['NP'])[99])
"""
"""
3.2 Evaluaciones simples y líneas base
"""
from nltk.corpus import conll2000
cp=nltk.RegexpParser("")
test_sents=conll2000.chunked_sents('test.txt',chunk_types=['NP'])
#print(cp.evaluate(test_sents))
"""
IOB tag accuracy indica que más de un tercio de las palabras están etiquetadas con O, no un NP chunk
Como no hay ningún chunk su precision, recall y f-measure son todos cero.
Probamos ahora con una expresion regular que busca etiquetas que empiezan con letras que son caracteristicas de nombres 
"""
grammar = r"NP: {<[CDJNP].*>+}"
cp = nltk.RegexpParser(grammar)
#print(cp.evaluate(test_sents))
"""
Aunque esto funciona se puede mejorar utilizando un corpus de entrenamiento para encontrar una tag que sea más probable para esa parte del texto
Es decir, usando un etiquetador unigrama para encontrar la tag del chunk teniendo la de las diferentes palabras que lo conforman
En el código a continuación convertimos de la representación en arbol a la IOB, la clase define un constructor y un parse 
"""
class UnigramChunker(nltk.ChunkParserI):
    def __init__(self,train_sents):
        train_data=[[(t,c) for w,t,c in nltk.chunk.tree2conlltags(sent)]
                      for sent in train_sents]
        self.tagger=nltk.UnigramTagger(train_data)
    def parse(self,sentence):
        pos_tags=[pos for (word,pos) in sentence]
        tagged_pos_tags=self.tagger.tag(pos_tags)
        chunktags=[chunktag for (pos,chunktag)in tagged_pos_tags]
        conlltags=[(word, pos, chunktag) for ((word,pos),chunktag)
                     in zip(sentence, chunktags)]
        return nltk.chunk.conllstr2tree(conlltags)
"""
El constructor espera oraciones de entrenamiento que vienen en forma de árbol
1. Convierte este datos con tree2conlltags a una lista de tripletas word,tag,chunk
2. Entrena con esos datos a un etiquetador de unigramas y lo guarda en self.tagger
3. Parse coje una sentencia etiquetada como entrada y extrae las etiquetas
4. Etiqueta las etiquetas de palabra con IOB chunk tags 
5. Extrae las etiquetas de chunks y combina con la oración original 
6. Finalmente lo vuelve a convertir a un arbol de chunks 
"""        
#Entrenamos en UnigramChunker con CoNLL 2000
test_sents=conll2000.chunked_sents('test.txt',chunk_types=['NP'])
train_sents=conll2000.chunked_sents('train.txt',chunk_types=['NP'])
unigram_chunker=UnigramChunker(train_sents)
#print(unigram_chunker.evaluate(test_sents))
#Da error de compilación...
postags=sorted(set(pos for sent in train_sents
                   for (word,pos) in sent.leaves()))
#print(unigram_chunker.tagger.tag(postags))

"""
3.3 Entrenando Chunkers basados en clasificadores
A veces las etiquetas de las palabras son insuficientes para determinar como una oración debería der ser chunqueada.
    Joey/NN sold/VBD the/DT farmer/NN rice/NN ./.
    Nick/NN broke/VBD my/DT computer/NN monitor/NN ./.
Estas dos tienen mismas tags de palabras pero distintos chunks
Para maximizar el rendimiento se tiene que tener en cuenta el contenido de las palabras
La manera de hacerlo es con un clasificador que chunquee la oración 
 
Primera clase es identidca a ConsecutivePosTagger pero llama a un extractor de características distinto y usa un 
Clasificador Maxent. La segunda clase es un contenedor alrededor del etiquetador que lo convierte en un chunker 
La segunda clase mapea los arboles de chunks en el corpus de entrenamiento a etiquetas de oraciones, parse convierte esto en chunk tree   
"""
class ConsecutiveNPChunkTagger(nltk.TaggerI):

    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = npchunk_features(untagged_sent, i, history) 
                train_set.append( (featureset, tag) )
                history.append(tag)
        self.classifier = nltk.MaxentClassifier.train( 
            train_set, algorithm='megam', trace=0)

    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = npchunk_features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)

class ConsecutiveNPChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        tagged_sents = [[((w,t),c) for (w,t,c) in
                         nltk.chunk.tree2conlltags(sent)]
                        for sent in train_sents]
        self.tagger = ConsecutiveNPChunkTagger(tagged_sents)

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w,t,c) for ((w,t),c) in tagged_sents]
        return nltk.chunk.conlltags2tree(conlltags)
"""
El extractor de características taggea el token actual y el chunker es similar al unigrama
"""
def npchunk_features(sentence,i,history):
    word,pos=sentence[i]
    if i==0:
        prevword,prevpos="<START>","<START>"
    else:
        prevword,prevpos=sentence[i-1]
    if i==len(sentence)-1:
        nextword,nextpos="<END>","<END>"
    else:
        nextword,nextpos=sentence[i+1]
    return {"pos":pos,
            "word":word,
            "prevpos":prevpos,
            "nextpos":nextpos,
            "prevpos+pos":"%s%s"%(prevpos,pos),
            "pos+nextpos":"%s%s"%(pos,nextpos),
            "tags-since-dt":tag_since_dt(sentence,i)}

def tag_since_dt(sentence,i):
    tags=set()
    for word,pos in sentence[:i]:
        if pos=='DT':
            tags=set()
        else:
            tags.add(pos)
    return '+'.join(sorted(tags))
#chunker = ConsecutiveNPChunker(train_sents)
#print(chunker.evaluate(test_sents))

"""
4. Recursión en Estructura Linguistica

4.1 Construyendo una estructura anidada con chunkers en cascada
Se puede construir estructuras de chunk de profundidad arbitraria creando una gramática de varias etapas que contiene reglas recursivas
Esta gramática tiene 4 fases: Sintagmas nominales, frases preposicionales, sintagmas verbales y oraciones. 
"""
grammar = r"""
  NP: {<DT|JJ|NN.*>+}          # Chunk sequences of DT, JJ, NN
  PP: {<IN><NP>}               # Chunk prepositions followed by NP
  VP: {<VB.*><NP|PP|CLAUSE>+$} # Chunk verbs and their arguments
  CLAUSE: {<NP><VP>}           # Chunk NP, VP
  """
cp = nltk.RegexpParser(grammar)
sentence = [("Mary", "NN"), ("saw", "VBD"), ("the", "DT"), ("cat", "NN"),
    ("sit", "VB"), ("on", "IN"), ("the", "DT"), ("mat", "NN")]
#print(cp.parse(sentence))
#No pilla el sintagma verbal encabezado por "saw"
#Probamos otra frase
sentence = [("John", "NNP"), ("thinks", "VBZ"), ("Mary", "NN"),
    ("saw", "VBD"), ("the", "DT"), ("cat", "NN"), ("sit", "VB"),
    ("on", "IN"), ("the", "DT"), ("mat", "NN")]
#print(cp.parse(sentence))
"""
La solución para estos problemas es hacer que el chunker vaya "loopeando" por sus patrones, tras probarlos todos repite el proceso 
Se añade un parámetro loop para especificar el numero de veces que se debe ejecutar eso
"""
cp=nltk.RegexpParser(grammar,loop=2)
#print(cp.parse(sentence))
"""
Este proceso permite crear estructuras más profundas pero debugearlas es dificil y es más sencillo encargarte del parseo
El proceso de cascada produce arboles de profundidad fija, no puede ser más que el número de fases y es insuficiente para un analisis sintactico completo   
"""
"""
4.2 Árboles
Un arbol es una serie de nodos etiquetados e interconectados, cada uno es alcanzable desde un único camino desde el nodo raíz
Se dice de los nodos que son padres, hijos y hermanos según su posición 

tree1=nltk.Tree('NP',['Alice'])
print(tree1)
tree2=nltk.Tree('NP',['the','rabbit'])
print(tree2)
tree3=nltk.Tree('VP',['chased',tree2])
tree4=nltk.Tree('NP',[tree1,tree3])
print(tree4)
print(tree4[1])
print(tree4[1].label())
print(tree4.leaves())
print(tree4[1][1][1])
tree3.draw()
"""
"""
4.3 Transversión de Arboles
Usamos función recursiva para recorrer un arbol 

def traverse(t):
    try:
        t.label()
    except AttributeError:
        print(t,end=" ")
    else:
        #Now we know that t.node is defined
        print('(',t.label(),end=" ")
        for child in t:
            traverse(child)
        print(')',end=" ")
#t = nltk.Tree('(S (NP Alice) (VP chased (NP the rabbit)))')
#traverse(t)
"""
"""
5. Reconocimiento de Entidades Nombradas
Una entidad nombrada es un sintagma nominal que se refiere a un tipo específico de individúo como organizaciones, personas, frases etc
Ejemplos de NEs
ORGANIZATION    Georgia-Pacific Corp., WHO
PERSON    Eddy Bonte, President Obama
LOCATION    Murray River, Mount Everest
DATE    June, 2008-06-29
TIME    two fifty a m, 1:30 p.m.
MONEY    175 million Canadian Dollars, GBP 10.40
PERCENT    twenty pct, 18.75 %
FACILITY    Washington Monument, Stonehenge
GPE    South East Asia, Midlothian

NER consiste en identificar todas las menciones textuales de entidades nombradas,se divide en dos tareas:
1 Identificar los límites de la NE 
2 Identificar su tipo
NER se usa como preludio para identificar relaciones en extracción de información 
Ejemplo:
Respondemos a la pregunta ¿Quien fue el primer presidente de los estados unidos? y el documento dado contiene esto
The Washington Monument is the most prominent structure in Washington, D.C. and one of the city's early attractions. It was built in honor of George Washington, who led the country to independence and then became its first President.
La respuesta se entiened que será "X fue el primer presidente de los estados unidos" donde X es un sintagma nominal y una entidad nobrada del tipo PERSON, NER nos indicaría cuales tienen el tipo correcto y cuales no
Una manera de identificar entidades nombradas sería buscar cada palabra en la lista de nombres apropiada , se podría usar un "gazetteer" o diccionario geográfico pero supone un problema porque no resuelve ambiguedades
Con nombres de personas u organizaciones también es dificil porque la covertura es peor y se actualiza a diario.
Otro problema son los nombres formados por varias palabras como "Stanford UNiversity" y nombres que contienen otros nombres 
Para NER se debe de poder identificar el comienzo y final de secuencias formadas por multiples tokens.
NER es una tarea apropiada para un acercamiento basado en clasificadores como los de chunking de sintagmas nominales, se puede hacer un etiquetador que use el formato IOB que permite este etiquetado
Eddy N B-PER
Bonte N I-PER
is V O
woordvoerder N O
van Prep O
diezelfde Pron O
Hogeschool N B-ORG
. Punc O
Con esta clasificación se puede entrenar un etiquetadory se puede convertir en un arbol de chunks con nltk.chunk.conlltags2tree()
NLTK tiene incorporado un clasificador que puede reconocer entidades nombradas con nltk.ne_chunk(), binary=True hace que las entidades nombradas se etiqueten con NE, si no está así el clasificador le añade PERSON, ORGANIZATION, and GPE.
Ejemplo de uso:
sent= nltk.corpus.treebank.tagged_sents()[22]
print(nltk.ne_chunk(sent, binary=True))
"""
sent= nltk.corpus.treebank.tagged_sents()[22]
print(nltk.ne_chunk(sent, binary=True))
print(nltk.ne_chunk(sent))

"""
6 Extracción de Relación
Se suele buscar las relaciones entre tipos específicos de entidades nombradas,
Una manera de hacer esto es buscando tripletas (X,a,Y) donde X e Y son entidades nombradas de los tipos requeridos y a es la cadena de caracteres que interviene entre X e Y.
Se usan expresiones regulares para obtener las instancias de a de la expresion que estamos buscando  
En este ejemplo se buscan strings que contengan la palabra in,
(?!\b.+ing\b) es una aserción negativa que nos permite ignorar strings como success in supervising the transition of, en la que in es seguida de un gerundio
"""
#Lo que definimos como IN es el patron que queremos buscar con la expresion regular
#En el primer for recorremos los documentos parseador del conjunto ieer 
#El segundo for extrae las tuplas formadas por la tag y el nombre con el in entre medias 
#metodo nltk.sem.extract_rels() es lo que usamos para obtenerlo
IN = re.compile(r'.*\bin\b(?!\b.+ing)')
for doc in nltk.corpus.ieer.parsed_docs('NYT_19980315'):
    for rel in nltk.sem.extract_rels('ORG','LOC',doc,corpus='ieer',pattern=IN):
        print(nltk.sem.rtuple(rel))
#Aún así puede haber falsos positivos al pillar palabras entre medias 
#Como también pillan las etiquetas de las palabras se puede tener en cuenta para las expresiones regulares
#El método clause() imprime las relaciones en forma de clausula donde el símbolo de relacion binaria se especifica como el valor del parámetro relsym
from nltk.corpus import conll2002
vnv= """
(
is/V|    # 3rd sing present and
was/V|   # past forms of the verb zijn ('be')
werd/V|  # and also present
wordt/V  # past of worden ('become)
)
.*       # followed by anything
van/Prep # followed by van ('of')
"""
VAN=re.compile(vnv,re.VERBOSE)
for doc in conll2002.chunked_sents('ned.train'):
    for rel in nltk.sem.extract_rels('PER','ORG',doc,corpus='conll2002',pattern=VAN):
        print(nltk.sem.clause(rel,relsym="VAN"))
        





