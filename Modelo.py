# Seccion 1 de codigo
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Desactiva los avisos y mensajes informativos

from tensorflow.keras.activations import relu
from tensorflow.keras.layers import Input
from tensorflow.keras import optimizers  # Para usar los optimizadores (calculo de gradientes descendente)
from tensorflow.keras.layers import Conv2D, MaxPooling2D  # Para crear las capas de la red neuronal
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.models import Sequential  # Para crear el modelo secuencial
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Para generar datos sinteticos
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# Seccion 2 de codigo
# Definir la ruta de las imagenes
entrenamiento = 'Entrenar/'
validacion = 'Validar/'

# Definir los hiperparametros de la arquitectura CNN
epocas = 20  # Numero de epocas
altura, anchura = 512, 512  # Dimensiones de las imagenes
batch_size = 20  # Tamaño del batch
pasos = 100  # Pasos por epoca

# Definir los hiperparametros de las capas convolucionales
kernel1 = 32
kernel1_size = (3, 3)
kernel2 = 64
kernel2_size = (3, 3)
size_pooling = (3, 3)  # Tamaño del max pooling

clases = 2  # Numero de clases

# Seccion 3 de codigo
# Generar datos sinteticos (ayuda cuando tenemos pocos datos de entrenamiento)

entrenar = ImageDataGenerator(
    rescale=1/255,
    zoom_range=0.3,
    horizontal_flip=True,
    rotation_range=30,  # Rotar las imágenes
    width_shift_range=0.2,  # Desplazar horizontalmente
    height_shift_range=0.2,  # Desplazar verticalmente
    brightness_range=[0.8, 1.2]  # Ajustar el brillo
)

validar = ImageDataGenerator(rescale=1/255)

# Leemos las imagenes
imagenes_entrenamiento = entrenar.flow_from_directory(entrenamiento, target_size=(altura, anchura),
                                                      batch_size=batch_size,
                                                      class_mode='categorical')  # Leemos las imagenes de entrenamiento
imagenes_validacion = validar.flow_from_directory(validacion, target_size=(altura, anchura), batch_size=batch_size,
                                                  class_mode='categorical')  # Leemos las imagenes de validacion

# Seccion 4 de codigo
# Construir arquitectura de la CNN

ModeloCNN = Sequential()  # Creamos el modelo secuencial

# Determinar las capas convolucionales
# Primera capa convolucional
# ModeloCNN.add(Conv2D(kernel1, kernel1_size, padding='same', input_shape=(altura, anchura, 3), activation=relu6))  # Capa convolucional
ModeloCNN.add(Input(shape=(altura, anchura, 3)))  # Define la forma de entrada
ModeloCNN.add(Conv2D(kernel1, kernel1_size, padding='same', activation=relu))  # Primera capa convolucional
# Agreagr un submuestreo
ModeloCNN.add(MaxPooling2D(pool_size=size_pooling))  # Max pooling

# Segunda capa convolucional
ModeloCNN.add(Conv2D(kernel2, kernel2_size, padding='same', activation=relu))  # Segunda capa convolucional
ModeloCNN.add(MaxPooling2D(pool_size=size_pooling))  # Max pooling
ModeloCNN.add(Flatten())  # Aplanar la salida de la capa convolucional. De matriz a vector grande

# Conectar la MLP (Red neuronal multicapa)
# Primera capa oculta
ModeloCNN.add(Dense(100, activation=relu))  # Capa densa
# Segunda capa oculta
ModeloCNN.add(Dense(200, activation=relu))  # Capa densa
ModeloCNN.add(Dropout(0.3))  # Dropout para evitar el sobreajuste
# Capa de salida
ModeloCNN.add(Dense(clases, activation='softmax'))  # Capa de salida

# Seccion 5 de codigo
# Establecer los parametros de entrenamiento
ModeloCNN.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', 'mse'])  # Compilamos el modelo

# Seccion 6 de codigo
# Entrenar el modelo
# Callback para detener el entrenamiento si no hay mejoras
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Callback para guardar el mejor modelo
model_checkpoint = ModelCheckpoint('Modelo/mejor_modelo.keras', save_best_only=True, monitor='val_loss')

# Entrenar el modelo con callbacks
ModeloCNN.fit(
    imagenes_entrenamiento,
    validation_data=imagenes_validacion,
    epochs=epocas,
    validation_steps=pasos,
    verbose=1,
    callbacks=[early_stopping, model_checkpoint]
)
# Seccion 7 de codigo
# Guardar el modelo
#ModeloCNN.save('Modelo/modelo.keras')  # Guardamos el modelo completo en el formato recomendado
ModeloCNN.save_weights('Modelo/modelo_pesos.weights.h5')  # Guardamos los pesos del modelo con la extensión correcta
