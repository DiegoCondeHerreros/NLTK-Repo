# -*- coding: utf-8 -*-
'''
Created on 1 mar. 2021

@author: DIEGO
'''
# CHULETA DE MANERAS DE RECORRER UN CONJUNTO
#for item in s    iterate over the items of s
#for item in sorted(s)    iterate over the items of s in order
#for item in set(s)    iterate over unique elements of s
#for item in reversed(s)    iterate over elements of s in reverse
#for item in set(s).difference(t)    iterate over elements of s not in t
from ast import Num

#Las funciones que en vez de terminar con un return terminan con un yield se les llama un generador, son funciones que devuelven
#un objeto tipo iterador.Cuando este listo para otra palabra continua la ejecución desde el yield.

#Funciones de alto orden: se puede pasar una funcion como primer parámetro de filter() para indicar que funcion actúa de filtro del conjunto que es el siguiente parámetro
#Se puede indicar los valores por defecto a la función para que en caso de que no se pasen haga lo que tiene que hacer
def repeat(msg='<empty>',num=1):
    return msg*num
print(repeat(num=3))
print(repeat(msg='Alice'))
print(repeat(num=5, msg='Alice'))

#Los parametros keyword deben ir a la derecha de los que no están nombrados
def generic(*args,**kwargs):
    print(args)
    print(kwargs)
generic(1,"African swallow",monty="python") 