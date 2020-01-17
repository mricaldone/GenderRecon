# GenderRecon
Red Neuronal que define el genero de una persona a partir de una imagen de su rostro (Formato: 50x50 en escala de grises).
Se la siguiente red neuronal https://github.com/mricaldone/RedNeuronal.
Es necesaria la instalcion de numpy y pil.

# Scripts
**entrenar.py** entrena la red a partir de un dataset de rostros aleatorios generados por StyleGAN2, clasificados manualmente y preprocesados en escala de grises 50x50 para su utilización en esta red.

**entrenar_errores.py** análogo a entrenar.py, sólo que entrena unicamente los casos erroneos. Por lo que el entrenamiento es exponencialmente más rápido.

**evaluar.py** pone a prueba la red utilizando un dataset de prueba generado por StyleGAN2.
