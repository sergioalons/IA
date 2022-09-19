import cv2

# Método para dibujar un rectángulo sobre un objeto detectado
def dibujar_rectangulo(img, clasificador, factorDeEscala, minVecinos, color, texto):
    # Convertimos la imagen a escala de grises
    imagen_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detectamos las características en la imagen de escala de grises y devolvemos las coordenadas, anchura y altura de lo detectado
    caracteristicas = clasificador.detectMultiScale(imagen_gris, factorDeEscala, minVecinos)
    coords = []
    # Dibujamos un rectángulo sobre las características detectadas y las etiquetamos
    for (x, y, w, h) in caracteristicas:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, texto, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    return coords


# Método para detectar todas las características
def detectar(img, caraCascade, ojoCascade, narizCascade, bocaCascade):
    # Dibujamos un rectángulo sobre la cara, si la detectamos nos devuelve un array de longitud 4
    coords = dibujar_rectangulo(img, caraCascade, 1.1, 10, (255, 0, 0), "Cara")
    # Si la característica es detectada llamamos al método dibujar_rectangulo y comparamos los parámetros que nos devuelve esta función
    # Si la longuitud del array es 4 entonces es que hemos detectado una cara, ahora pasamos a dibujar dentro de ella el resto de características
    if len(coords) == 4:
        # Actualizamos la región de interés recortando la imagen
        rdi_img = img[coords[1]:coords[1] + coords[3], coords[0]:coords[0] + coords[2]]
        # Le pasamos la región de interés, el clasificador, un factor de escala, el número mínimo de vecinos, el color y el texto a etiquetar
        coords = dibujar_rectangulo(rdi_img, ojoCascade, 1.3, 12, (0, 0, 255), "Ojo")
        coords = dibujar_rectangulo(rdi_img, narizCascade, 1.3, 4, (0, 255, 0), "Nariz")
        coords = dibujar_rectangulo(rdi_img, bocaCascade, 1.1, 20, (255, 255, 255), "Boca")
    return img


# Cargamos los clasificadores de características
caraCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
ojoCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
narizCascade = cv2.CascadeClassifier('Nariz.xml')
bocaCascade = cv2.CascadeClassifier('Mouth.xml')

# También podemos capturarlo de una WebCam en tiempo real, para esto le introducimos como parámetros 0 o -1 dependiendo del tipo de WebCam que poseamos
#video_capture = cv2.VideoCapture(-1)

# Capturamos el vídeo a partir de un archivo
video_capture = cv2.VideoCapture('video.mp4')

while True:
    # Leémos la imagen a partir del streaming del vídeo
    _, img = video_capture.read()
    # Llamamos al método detectar que hemos definido arriba
    img = detectar(img, caraCascade, ojoCascade, narizCascade, bocaCascade)
    # Escribimos la imagen procesada en una pestaña nueva
    cv2.imshow("Detector de rostros", img)
    # Para terminar el programa basta con presionar la letra 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberamos la Webcam o vídeo
video_capture.release()
# Eliminamos la ventana que hemos abierto
cv2.destroyAllWindows()
