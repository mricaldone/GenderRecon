from PIL import Image
import numpy as np
from NeuralNetwork.RedNeuronal import *
from NeuralNetwork.Funciones import *

#ORIGINAL
rn = RedNeuronal(2500, [2500,70,2], Sigmoide())
rn.cargar('genders')

#print(rn.capas[1].matriz_w)
#print(rn.capas[1].matriz_w[0,0:])

def obtener_vector_de_pesos(capa, neurona):
	return rn.capas[capa].matriz_w[neurona,0:].tolist()[0]

def imprimir_grafico(matriz, minval, maxval):
	n = len(matriz[0])
	#TRANSFORMACION LINEAL y = ax + b
	a = 255/(maxval-minval)
	b = -a * minval
	#IMPRIMIR
	paper = np.zeros((n, n, 3), dtype=np.uint8)
	for i in range(n):
			for j in range(n):
				val = matriz[i][j]
				rgb = int(a * val + b)
				pixel = [rgb, 0, 0]
				paper[i,j] = pixel
	img = Image.fromarray(paper, 'RGB')
	img.transpose(Image.FLIP_TOP_BOTTOM).show()
	
pesos = obtener_vector_de_pesos(0,0)
maxval = np.amax(pesos)
minval = np.amin(pesos)

n = 50
matriz = []
for i in range(n):
	fila = pesos[i * n:i * n + n]
	matriz.append(fila)
#print(matriz)

imprimir_grafico(matriz, maxval, minval)


input('Terminar...')
