from PIL import Image
import os

while True:
	dir_name = 'faces'
	file_list = os.listdir(dir_name)
	file_name = file_list[0]
	img = Image.open(dir_name + '/' + file_name)
	img.show()
	rta = input('Es hombre? (S/N/X): ')
	if rta.upper() == 'S':
		os.rename(dir_name + '/' + file_name, 'hombres/' + file_name)
	if rta.upper() == 'N':
		os.rename(dir_name + '/' + file_name, 'mujeres/' + file_name)
	if rta.upper() == 'X':
		os.remove(dir_name + '/' + file_name)


