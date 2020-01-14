from PIL import Image
import numpy as np
from NeuralNetwork.RedNeuronal import *
from NeuralNetwork.Funciones import *

def imprimir_grafico(rn, paso, cant_decimales):
	#w, h = 512, 512
	#data = np.zeros((h, w, 3), dtype=np.uint8)
	#data[256, 256] = [255, 0, 0]
	#img = Image.fromarray(data, 'RGB')
	#img.save('my.png')
	#img.show()
	x = 0
	y = 0
	dim = int(1/paso) + 1
	#DEFINO EL TAMAÑO DEL PAPEL
	w, h = dim, dim
	data = np.zeros((h, w, 3), dtype=np.uint8)
	for i in range(dim):
		x = 0
		for j in range(dim):
			r = rn.procesar([x,y])[0]
			pixel = [int(255 * r), 0, 0]
			data[i,j] = pixel
			x = x + paso
		y = y + paso
	img = Image.fromarray(data, 'RGB')
	img.transpose(Image.FLIP_TOP_BOTTOM).show()

#ORIGINAL
rn = RedNeuronal(2500, [2500,70,2], Sigmoide())
#rn.cargar('genders')

#print(rn.capas[1].matriz_w)
#print(rn.capas[1].matriz_w[0,0:])

pesos = rn.capas[1].matriz_w[0,0:].tolist()

print(type(pesos))
print(pesos.size)

maxval = np.amax(pesos)
minval = np.amin(pesos)
print(len(pesos))

n = 50
matriz = []
for i in range(n):
	fila = pesos[i * n:i * n + n]
	print(len(fila))
	matriz.append(fila)
#print(matriz)

input('Terminar...')
