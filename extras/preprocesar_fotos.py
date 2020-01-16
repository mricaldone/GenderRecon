import os
import cv2
import numpy as np

IMG_W = 50
IMG_H = 50
DIR_IN = 'fotos'
DIR_OUT = 'faces'

def procesar_imagenes(file_in, file_out):
	face_orig = img = cv2.imread(DIR_IN + '/' + file_in, 0)
	#DETECCION DE OJOS Y ROSTRO
	face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
	#eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
	faces_detected = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
	(x, y, w, h) = faces_detected[0]
	cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1);
	#RECORTAR ROSTRO
	crop_img = img[y:y+h, x:x+w]
	#REDIMENSIONAR IMAGEN
	out_img = cv2.resize(crop_img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
	#NORMALIZAR IMAGEN
	norm_img = np.zeros((IMG_W, IMG_H))
	norm_img = cv2.normalize(out_img, norm_img, 0, 255, cv2.NORM_MINMAX)
	#GRABAR IMAGEN
	cv2.imwrite(DIR_OUT + '/' + file_out, norm_img)

file_list = os.listdir(DIR_IN)
list_len = len(file_list)
for i,file_name in enumerate(file_list):
	os.system('clear')
	print('ARCHIVO',file_name)
	print('RESTANTE', i+1, '/', list_len)
	try:
		procesar_imagenes(file_name, str(i) + '.jpg')
	except:
		print('ERROR',file_name)

input('LISTO!')
