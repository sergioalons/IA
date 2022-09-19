### Sistema de detección de caras, nariz, boca y ojos en rostros hecho en OpenCV

Todos los archivos de datos de los clasificadores de las diferentes partes han sido obtenidos de 
https://github.com/kipr/opencv/tree/master/data/haarcascades

El código está comentado en cada una de las partes importantes explicando la funcionalidad sobre la misma.
 
Así mismo para su funcionamiento es necesaria únicamente la librería de OpenCV para python, pero debido a problemas con 
la última versión se recomienda instalar la versión 4.5.4.60 en caso de que la última de fallos.

El algoritmo usado para detectar los rostros de denomina 'Haar Cascade'. Es un algoritmo de detección de objetos que se 
utiliza para identificar rostros en una imagen o un video en tiempo real.
El algoritmo recibe muchas imágenes positivas que consisten en rostros y muchas imágenes negativas que no constan de 
ningún rostro para entrenar sobre ellas.

Toda la información sobre este algoritmo aquí escrita ha sido sacada de aquí y si se quiere profundizar sobre el tema en
esa página web vienen detalles técnicos sobre el algoritmo en cuestión
https://towardsdatascience.com/face-detection-with-haar-cascade-727f68dafd08

Notas adicionales:
En el caso de que el vídeo detecte mal alguna característica o esté duplicada habría que ir cambiando el factor de 
escala para ajustarlo hasta que detecte bien la característica.