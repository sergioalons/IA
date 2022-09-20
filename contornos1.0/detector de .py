import numpy as np
import cv2
import random as rng

#Leemos y mostramos la imagen original en color
im= cv2.imread('formas.png',cv2.IMREAD_COLOR)
cv2.imshow('0 color', im)
cv2.waitKey()

#Leemos y mostramos la imagen original en niveles de gris
imgray= cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
cv2.imshow('0 gris',imgray)
cv2.waitKey()

#Leemos y mostramos la imagen de contornos detectados utilizando Canny
canny=cv2.Canny(imgray,127,254)
cv2.imshow('CANNY',	canny)
cv2.waitKey()

#obtenemos los datos de los contornos y su jerarquia
contornos, jerarquia = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

#y los mostramos sobre una imagen 
imdraw=cv2.imread('negro.png',cv2.IMREAD_COLOR)

	
cv2.drawContours(imdraw,contornos,0,(20,150,200),2)
cv2.drawContours(imdraw,contornos,1,(100,100,0),2)
cv2.drawContours(imdraw,contornos,2,(256,256,256),2)
cv2.drawContours(imdraw,contornos,3,(250,10,200),2)
	
cv2.imshow('contornos',imdraw)
cv2.waitKey()

#num contornos detectados
numContornos= len(contornos)
print("num contornos detectados es",numContornos)

#Jerarquia[Next,Previous,First_Child, Parent]
print(jerarquia)

#obtenemos y mostramos los momentos de cada uno de los contornos
#y su centro para reconocer su posicion en la imagen
#tambien el area y el perimetro
for contorno in contornos:
	M=cv2.moments(contorno)
	print(M)
	cx=int(M['m10']/M['m00'])
	cy=int(M['m01']/M['m00'])
	HU=cv2.HuMoments(cv2.moments(contorno))
	print(HU)
	print('El centro de gravedad del contorno es (',cx,',',cy,')')
	area= cv2.contourArea(contorno)
	print('El area es ',area) 
	perimeter=cv2.arcLength(contorno,True)
	print('El perimetro es ',perimeter)