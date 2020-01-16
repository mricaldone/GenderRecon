#import pygame
import random
import os
from PIL import Image
import numpy as np

from NeuralNetwork.RedNeuronal import *
from NeuralNetwork.Funciones import *

#MEJOR RESULTADO LR=0.5 E=20 M=200

LEARNING_RATE = 0.5							#RATIO DE APRENDIZAJE
EPOCHS = 50									#CANTIDAD DE VECES QUE SE REPITE CADA LOTE O DATASET
SAMPLES = 1000								#CANTIDAD DE MUESTRAS DE CADA CLASE
TESTS = 0.2									#PORCENTAJE DE MUESTRAS UTILIZADAS PARA EL TEST
MALE_DATASET = 'datasets/random/males'		#DIRECTORIO CON FOTOS DE HOMBRES
FEMALE_DATASET = 'datasets/random/females'	#DIRECTORIO CON FOTOS DE MUJERES

#DEFINICION DE LA ESTRUCTURA DE LA RED
rn = RedNeuronal(2500, [2500,71,2], Sigmoide())

def cargar_rostros(dir_name, label):
	rostros = []
	file_list = os.listdir(dir_name)
	for file_name in file_list:
		file_path = dir_name + '/' + file_name
		rostro = np.matrix(Image.open(file_path).convert('L')).A1
		rostro = np.divide(rostro, 255)
		rostros.append((rostro,label,file_path))
	random.shuffle(rostros)
	return rostros

def probar_red(rostros):
	aciertos = 0
	for i,rostro in enumerate(rostros):
		imagen = rostro[0]
		etiqueta = rostro[1]
		resultado = rn.procesar(imagen)
		if resultado[0] > resultado[1]:
			if etiqueta[0] == 1:
				aciertos += 1
		else:
			if etiqueta[1] == 1:
				aciertos += 1
	print('Porcentaje de aciertos:',round(100*aciertos/(i+1),2),'%')
	
#print('CARGANDO RED...')
#rn.cargar('genders')

print('CARGANDO ROSTROS MASCULINOS...')
rostros_masculinos = cargar_rostros(MALE_DATASET, [1,0])

print('CARGANDO ROSTROS FEMENINOS...')
rostros_femeninos = cargar_rostros(FEMALE_DATASET, [0,1])

print('PREPARANDO ROSTROS...')
tam = min(len(rostros_masculinos), len(rostros_femeninos), SAMPLES)
rostros = rostros_femeninos[:tam] + rostros_masculinos[:tam]
random.shuffle(rostros)
cut = int(len(rostros) * TESTS)
rostros_test = rostros[:cut]
rostros_train = rostros[cut:]

print('ENTRENANDO RED...')
errores = 0
total = 0
for e in range(EPOCHS):
	probar_red(rostros_test)
	for i,rostro in enumerate(rostros_train):
		imagen = rostro[0]
		etiqueta = rostro[1]
		resultado = rn.procesar(imagen)
		if resultado[0] > resultado[1]:
			#ES HOMBRE
			if etiqueta[0] != 1:
				errores += 1
		else:
			#ES MUJER
			if etiqueta[1] != 1:
				errores += 1
		total += 1
		print('EPOCH:',str(e + 1) + '/' + str(EPOCHS),' ',end='')
		print('SAMPLE:',str(i + 1) + '/' + str(len(rostros_train)),' ',end='')
		print('ERROR:',round(100*errores/total,2),'%',' ',end='')
		print(' ' * 10, end='\r')
		rn.entrenar(imagen, etiqueta)
	print('')
print(' ' * 60, end='\r')
print('')

print('GUARDANDO RED...')
rn.guardar('genders')

print('PROBANDO RED...')
probar_red(rostros_test)

input('FINALIZADO...')
#pygame.init() 

#white = (255, 255, 255)
#black = (0, 0, 0)
 
#X = 100
#Y = 100
 
#display_surface = pygame.display.set_mode((X, Y )) 
#pygame.display.set_caption('GenderRecon') 
#image = pygame.image.load(r'faces/1.jpg')
#font = pygame.font.Font('freesansbold.ttf', 12)

#while True:
#	file_path, tipo = obtener_imagen_aleatoria()
#	img_vector = np.matrix(Image.open(file_path).convert('L')).A1
#	img_vector = np.divide(img_vector,255)
#	resultado = rn.procesar(img_vector)
	#GUI
#	display_surface.fill(black) 
#	display_surface.blit(image, (25, 0)) 
	
#	val1 = round(resultado[0] * 100,2)
#	text1 = font.render("Rasgos masculinos: " + str(val1), True, white, black)
#	textRect1 = text1.get_rect()
#	textRect1.center = (X // 2, 60)
#	display_surface.blit(text1, textRect1)
	
#	val2 = round(resultado[1] * 100,2)
#	text2 = font.render("Rasgos femeninos: " + str(val2), True, white, black)
#	textRect2 = text2.get_rect()
#	textRect2.center = (X // 2, 80)
#	display_surface.blit(text2, textRect2)
	
#	for event in pygame.event.get() :  
#		if event.type == pygame.QUIT : 
#			pygame.quit() 
#			quit()   
#		pygame.display.update()
	
#	input("Continuar...")
