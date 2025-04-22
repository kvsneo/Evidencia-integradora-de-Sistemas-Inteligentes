import os
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import numpy as np
import cv2  # Biblioteca para capturar imágenes desde la cámara
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model

# Configuración inicial
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Desactiva los avisos y mensajes informativos

# Dimensiones de las imágenes
altura, anchura = 512, 512

# Carga el modelo y los pesos
modelo = 'Modelo/mejor_modelo.keras'
pesos = 'Modelo/modelo_pesos.weights.h5'

ModeloCNN = load_model(modelo)  # Cargamos el modelo
ModeloCNN.load_weights(pesos)  # Cargamos los pesos

def clasificar_imagen(ruta_imagen):
    imagen = load_img(ruta_imagen, target_size=(altura, anchura))  # Cargamos la imagen
    imagen = img_to_array(imagen)  # Convertimos la imagen a un array
    imagen = np.expand_dims(imagen, axis=0)  # Expandimos las dimensiones de la imagen

    prediccion = ModeloCNN.predict(imagen)  # Hacemos la predicción
    max = np.argmax(prediccion)

    if max == 0:
        resultado = "Es muy probable que sea una imagen generada por IA"
    else:
        resultado = "Es muy probable que sea una imagen real"
    return resultado

# Función para seleccionar una imagen
def seleccionar_imagen():
    ruta_imagen = filedialog.askopenfilename(filetypes=[("Archivos de imagen", "*.jpg;*.png;*.jpeg")])
    if ruta_imagen:
        mostrar_y_clasificar(ruta_imagen)

# Función para tomar una foto con la cámara
def tomar_foto():
    camara = cv2.VideoCapture(0)  # Abrir la cámara
    if not camara.isOpened():
        etiqueta_resultado.config(text="No se pudo acceder a la cámara")
        return

    ret, frame = camara.read()  # Capturar un cuadro
    if ret:
        ruta_foto = "foto_capturada.jpg"
        cv2.imwrite(ruta_foto, frame)  # Guardar la foto capturada
        camara.release()
        cv2.destroyAllWindows()
        mostrar_y_clasificar(ruta_foto)
    else:
        etiqueta_resultado.config(text="No se pudo capturar la foto")
        camara.release()
        cv2.destroyAllWindows()

# Función para mostrar la imagen y clasificarla
def mostrar_y_clasificar(ruta_imagen):
    # Mostrar la imagen seleccionada o capturada
    img = Image.open(ruta_imagen)
    img = img.resize((250, 250))  # Redimensionar para mostrar en la interfaz
    img = ImageTk.PhotoImage(img)
    panel_imagen.config(image=img)
    panel_imagen.image = img

    # Clasificar la imagen
    resultado = clasificar_imagen(ruta_imagen)
    etiqueta_resultado.config(text=resultado)

# Crear la interfaz gráfica
ventana = tk.Tk()
ventana.title("Veritas - Clasificador de Imágenes")
ventana.geometry("500x600")
ventana.configure(bg="#f5f5f5")  # Fondo claro

# Título principal
titulo = Label(ventana, text="Veritas", font=("Arial", 24, "bold"), bg="#f5f5f5", fg="#333")
titulo.pack(pady=10)

# Subtítulo
subtitulo = Label(ventana, text="Clasificador de imágenes IA vs Real", font=("Arial", 14), bg="#f5f5f5", fg="#555")
subtitulo.pack(pady=5)

# Botón para seleccionar imagen
boton_seleccionar = Button(ventana, text="Seleccionar Imagen", command=seleccionar_imagen, font=("Arial", 12), bg="#4CAF50", fg="white", relief="flat", padx=10, pady=5)
boton_seleccionar.pack(pady=10)

# Botón para tomar foto
boton_tomar_foto = Button(ventana, text="Tomar Foto", command=tomar_foto, font=("Arial", 12), bg="#2196F3", fg="white", relief="flat", padx=10, pady=5)
boton_tomar_foto.pack(pady=10)

# Panel para mostrar la imagen seleccionada o capturada
panel_imagen = Label(ventana, bg="#f5f5f5")
panel_imagen.pack()

# Etiqueta para mostrar el resultado
etiqueta_resultado = Label(ventana, text="", font=("Arial", 14), bg="#f5f5f5", fg="#333")
etiqueta_resultado.pack(pady=20)

# Iniciar la interfaz gráfica
ventana.mainloop()