import random
import os
import numpy as np

from PIL import Image
from NeuralNetwork.RedNeuronal import *

MALE_DATASET = 'datasets/testing/males'
FEMALE_DATASET = 'datasets/testing/females'

rn = RedNeuronal(2500, [2500,71,2], Sigmoide())

def cargar_rostros(dir_name, label):
	rostros = []
	file_list = os.listdir(dir_name)
	for file_name in file_list:
		full_path = dir_name + '/' + file_name
		rostro = np.matrix(Image.open(full_path).convert('L')).A1
		rostro = np.divide(rostro, 255)
		rostros.append((rostro,label,full_path))
	return rostros

print('CARGANDO RED...')
rn.cargar('genders.rn')

print('CARGANDO ROSTROS DE PRUEBA...')
rostros_masculinos = cargar_rostros(MALE_DATASET, None)
rostros_femeninos = cargar_rostros(FEMALE_DATASET, None)
rostros = rostros_masculinos + rostros_femeninos
random.shuffle(rostros)

print('PROBANDO RED...')
for rostro in rostros:
	imagen = rostro[0]
	full_path = rostro[2]
	resultado = rn.procesar(imagen)
	print("Rasgos masculinos:",round(resultado[0] * 100,2),"%")
	print("Rasgos femeninos:",round(resultado[1] * 100,2),"%")
	if resultado[0] > resultado[1]:
		print('Veredicto: Es hombre')
	else:
		print('Veredicto: Es mujer')
	img = Image.open(full_path)
	img.show()
	input('Continuar...')
