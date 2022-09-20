import cv2
import numpy as np 

#Leemos la imagen original
img = cv2.imread('sudoku.jpg',cv2.IMREAD_COLOR)

#Convertimos la imagen a niveles de gris
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#Encontramos los contornos empleando canny
edges = cv2.Canny(gray,50,200)

#Detectamos las lineas empleando la transforma de Hough
lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength= 10,maxLineGap=250)

#Mostramos el numero de lineas encontrado
print("num de lineas encontradas : %r" %len(lines))

#Mostramos el array de 3 dimensiones devuelto
print(lines)


#Dibujamos las lineas sobre la imagen original
for line in lines:
    x1,y1,x2,y2=line[0]
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),1)
#Mostramos la imagen con las lineas detectadas
cv2.imshow("Result image",img)
cv2.waitKey()