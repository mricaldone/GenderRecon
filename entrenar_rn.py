#import pygame
import random
import os
from PIL import Image
import numpy as np

from NeuralNetwork.RedNeuronal import *
from NeuralNetwork.Funciones import *

LEARNING_RATE = 0.5
EPOCHS = 2000

f = Sigmoide()
#FUNCIONA MUY BIEN PARA 10000 EPOCAS 0.5 LR (OVERFITTING?)
#rn = RedNeuronal(2500, [2500,2,1], f)
#FUNCIONA MUY BIEN PARA 10000 EPOCAS 0.5 LR (OVERFITTING?)
#rn = RedNeuronal(2500, [2500,70,2,1], f)
#FUNCIONA MUY MAL PARA 1000 EPOCAS 0.5 LR
#rn = RedNeuronal(2500, [2500,242,22,2,1], f)

rn = RedNeuronal(2500, [2500,70,2], f)
rn.cargar('genders')

def obtener_imagen_aleatoria():
	r1 = random.randint(0,1)
	if r1 == 1:
		dir_name = 'hombres'
		tipo = [1,0]
	if r1 == 0:
		dir_name = 'mujeres'
		tipo = [0,1]
	file_list = os.listdir(dir_name)
	r2 = random.randint(0,len(file_list) - 1)
	file_name = file_list[r2]
	file_path = dir_name + '/' + file_name
	return file_path, tipo

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
	rn.entrenar(img_vector, tipo)
	#input('CONTINUAR...')

print('GUARDANDO RED...')
rn.guardar('genders')

print('PROBANDO RED...')
while True:
	file_path, tipo = obtener_imagen_aleatoria()
	img_vector = np.matrix(Image.open(file_path).convert('L')).A1
	img_vector = np.divide(img_vector,255)
	resultado = rn.procesar(img_vector)
	print("Rasgos masculinos:",round(resultado[0] * 100,2),"%")
	print("Rasgos femeninos:",round(resultado[1] * 100,2),"%")
	if resultado[0] > resultado[1]:
		print('Es hombre')
	else:
		print('Es mujer')
	img = Image.open(file_path)
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
