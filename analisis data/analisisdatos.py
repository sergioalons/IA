import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



print('Bienvenido, por favor para realizar predicción  introduzca los siguientes datos:')
print('Introduce el año de los datos(2016 - 2020), si quiere todos los datos introduzca un espacio:')
datos_elegidos = input()
print('Introduce el número de días:')
numdias = int(input())
print('Introduce la columna a predecir (1 para Último,2 para Apertura,3 para Máximo y 4 para Mínimo):')
columna = int(input())
print('Introduce el numero de epochs:')
e = int(input())
 
if (datos_elegidos == "2016"):
    dele = 'Datos2016.csv'
elif(datos_elegidos == "2017"):
    dele = 'Datos2017.csv'
elif(datos_elegidos == "2018"):
    dele = 'Datos2018.csv'
elif(datos_elegidos == "2019"):
    dele = 'Datos2019.csv'
elif(datos_elegidos == "2020"):
    dele = 'Datos2020.csv'
else :
    dele = 'DatosTodos.csv'

dataset_train = pd.read_csv(dele)


if columna==1:
    campo="Último"
elif columna==2:
    campo="Apertura"
elif  columna==3:
    campo="Máximo"
elif  columna==4:
    campo="Mínimo"

training_set = dataset_train.iloc[:,columna:columna+1].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_escalado =sc.fit_transform(training_set) 
# estructura com 60 pasos temporales y 1 salida
X_train=[]
Y_train=[]
for i in range(60,255):
    X_train.append(training_set_escalado[i-60:i,0])
    Y_train.append(training_set_escalado[i,0])
X_train,Y_train= np.array(X_train),np.array(Y_train)

# poner más argumentos
X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
#modelo rnw
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import tensorflow as tf

regresor = Sequential()
#agregar primera capa y reg de desercion
regresor.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
regresor.add(Dropout(0.2))
# agregar 2 capa
regresor.add(LSTM(units=50,return_sequences=True))
regresor.add(Dropout(0.2))
# agregar 3 capa
regresor.add(LSTM(units=50,return_sequences=True))
regresor.add(Dropout(0.2))
# agregar 4 capa
regresor.add(LSTM(units=50))
regresor.add(Dropout(0.2))
#capa de salida
regresor.add(Dense(units=1))
#optimizar y funcion de perdida(compilacion)
regresor.compile(optimizer = 'adam',loss='mean_squared_error')
#encajar red neuronal en set de entrenamiento
regresor.fit(X_train,Y_train,epochs=e,batch_size=32)

# 3 conseguir datos reales
dataset_test= pd.read_csv(dele)
test_set = dataset_test.iloc[:,columna:columna+1].values
#predecir 
dataset_total = pd.concat((dataset_train[campo],dataset_test[campo]),axis=0)
inputs = dataset_total[len(dataset_total)- len(dataset_test) -60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs) 

#agregar dimension extra dandole buena estructura
X_test=[]
for i in range(60,60+numdias): 
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
#prediccion
prediccion_precio = regresor.predict(X_test)
prediccion_precio = sc.inverse_transform(prediccion_precio)

#visualizacion
plt.plot(test_set,color='red',label='Prediccion accion real')
plt.plot(prediccion_precio,color='blue',label='Prediccion ')
plt.title('prediccion ibex')
plt.xlabel('tiempo')
plt.ylabel('precio')
plt.legend()
plt.show()