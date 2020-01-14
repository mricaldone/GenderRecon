#import pygame
import random
import os
from PIL import Image
import numpy as np

from NeuralNetwork.RedNeuronal import *
from NeuralNetwork.Funciones import *

LEARNING_RATE = 0.5
EPOCHS = 10

rn = RedNeuronal(2500, [2500,70,2], Sigmoide())

#print('CARGANDO RED...')
#rn.cargar('genders')

def cargar_rostros(dir_name, label):
	rostros = []
	file_list = os.listdir(dir_name)
	for file_name in file_list:
		rostro = np.matrix(Image.open(dir_name + '/' + file_name).convert('L')).A1
		rostro = np.divide(rostro, 255)
		rostros.append((rostro,label,file_name))
	return rostros

print('CARGANDO ROSTROS MASCULINOS...')
rostros_masculinos = cargar_rostros('hombres', [1,0])

print('CARGANDO ROSTROS FEMENINOS...')
rostros_femeninos = cargar_rostros('mujeres', [0,1])

print('PREPARANDO ROSTROS')
tam = min(len(rostros_masculinos), len(rostros_femeninos))
rostros = rostros_femeninos[:tam] + rostros_masculinos[:tam]
random.shuffle(rostros)

print('ENTRENANDO RED...')
total = EPOCHS * len(rostros)
for e in range(EPOCHS):
	for i,rostro in enumerate(rostros):
		actual = e * len(rostros) + (i + 1)
		print(str(actual) + '/' + str(total), end='\r')
		imagen = rostro[0]
		label = rostro[1]
		rn.entrenar(imagen, label)

#print('GUARDANDO RED...')
#rn.guardar('genders')

print('CARGANDO ROSTROS DE PRUEBA...')
rostros = cargar_rostros('faces', None)
random.shuffle(rostros)

print('PROBANDO RED...')
for rostro in rostros:
	imagen = rostro[0]
	file_name = rostro[2]
	resultado = rn.procesar(imagen)
	print("Rasgos masculinos:",round(resultado[0] * 100,2),"%")
	print("Rasgos femeninos:",round(resultado[1] * 100,2),"%")
	if resultado[0] > resultado[1]:
		print('Es hombre')
	else:
		print('Es mujer')
	img = Image.open('faces/' + file_name)
	img.show()
	input('Continuar...')

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
