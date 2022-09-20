import cv2
import numpy as np 

#Leemos la imagen original
img = cv2.imread('circles.png',cv2.IMREAD_COLOR)

#Convertimos la imagen a niveles de gris
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#Encontramos los contornos empleando canny
edges = cv2.Canny(gray,50,200)

#Detectamos las circuferencias empleando la transforma de Hough


circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1.2, 100)
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
    cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)
cv2.imshow('circulos detectados',img)
cv2.waitKey(0)
