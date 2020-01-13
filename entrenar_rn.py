import random
import os
from PIL import Image
import numpy as np

#from Funciones import *
from NeuralNetwork.RedNeuronal import *
from NeuralNetwork.Funciones import *

LEARNING_RATE = 0.5
EPOCHS = 10000

f = Sigmoide()
#FUNCIONA MUY BIEN PARA 10000 EPOCAS 0.5 LR (OVERFITTING?)
#rn = RedNeuronal(2500, [2500,2,1], f)
#FUNCIONA MUY BIEN PARA 10000 EPOCAS 0.5 LR (OVERFITTING?)
#rn = RedNeuronal(2500, [2500,70,2,1], f)
#FUNCIONA MUY MAL PARA 1000 EPOCAS 0.5 LR
#rn = RedNeuronal(2500, [2500,242,22,2,1], f)

rn = RedNeuronal(2500, [2500,70,2,1], f)
rn.cargar('genders')

def obtener_imagen_aleatoria():
	r1 = random.randint(0,1)
	if r1 == 1:
		dir_name = 'hombres'
	if r1 == 0:
		dir_name = 'mujeres'
	file_list = os.listdir(dir_name)
	r2 = random.randint(0,len(file_list) - 1)
	file_name = file_list[r2]
	file_path = dir_name + '/' + file_name
	return file_path, r1

print('ENTRENANDO RED...')
for i in range(EPOCHS):
	print(str(i + 1) + '/' + str(EPOCHS), end='\r')
	#ABRE DE MANERA ALEATORIA UNA IMAGEN DEL DIRECTORIO DE HOMBRES O MUJERES
	file_path, tipo = obtener_imagen_aleatoria()
	#CONVIERTE LA IMAGEN A VECTOR
	img_vector = np.matrix(Image.open(file_path).convert('L')).A1
	#NORMALIZA LA IMAGEN
	img_vector = np.divide(img_vector,255)
	#ENTRENA LA RED
	rn.entrenar(img_vector, [tipo])
	#input('CONTINUAR...')

print('GUARDANDO RED...')
rn.guardar('genders')

print('PROBANDO RED...')
while True:
	file_path, tipo = obtener_imagen_aleatoria()
	img_vector = np.matrix(Image.open(file_path).convert('L')).A1
	img_vector = np.divide(img_vector,255)
	resultado = rn.procesar(img_vector)[0]
	print(resultado)
	if round(resultado,0) == 1:
		print('Es hombre',round((resultado - 0.5) * 200, 1),'%')
	else:
		print('Es mujer' ,round((0.5 - resultado) * 200, 1),'%')
	if round(resultado,0) == tipo:
		print('Correcto')
	else:
		print('Error')
	img = Image.open(file_path)
	img.show()
	input('Continuar...')


