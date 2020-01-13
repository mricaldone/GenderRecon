# GenderRecon
Red Neuronal que define el genero de una persona a partir de una imagen de su rostro (Formato: 50x50 en escala de grises).

# Scripts
<b>preprocesar_fotos.py</b> extrae todos los rostros encontrados en todas las fotos de la carpeta ''fotos'' y los guarda en formato 50x50 escala de grises en la carpeta ''faces''.

<b>clasificador.py</b> muestra las fotos de la carpeta ''faces'' una a una y permite al usuario definir si el rostro corresponde a una mujer o a un hombre. En el caso de corresponder a una mujer la foto se mueve a la carpeta ''mujeres'' y en caso de ser hombre a ''hombres''.

<b>entrenar_rn.py</b> entrena a la red neuronal a partir de las imagenes encontradas en las carpetas ''mujeres'' y ''hombres''. Genera el archivo genders.dat que puede ser utilizado para volver a cargar los pesos de la red neuronal.
