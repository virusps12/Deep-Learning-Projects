import pandas as pd
import numpy as np
import tensorflow as tf

df = pd.read_csv("Churn_Modelling.csv")
X = df.iloc[: , 3:-1].values
y = df.iloc[: , -1].values

#Label Encoding the Gender Column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[: , 2] = le.fit_transform(X[: , 2])

#One Hot Encoding The Country Column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
X = np.array(ct.fit_transform(X))

#Splitting The Data to train set and test set

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=1)

#Feature Scalling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#Building The ANN

#Initialising The ANN
ann = tf.keras.models.Sequential()

#Adding The Input Layer and hidden layer
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))
#Adding The 2nd hidden layer
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))
#Adding The Outpurt Layer
ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))

#Training The Ann

#compiling the ANN
ann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#training the ANN
ann.fit(X_train,y_train,batch_size=32,epochs=100)

#Predicting The Test Set result

print(ann.predict(sc.transform([[1 , 0 , 0 , 600, 1 , 40 , 3, 60000 , 2 , 1 , 1, 50000]]))>0.5)

#Predicting The Test Set result
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

#Making The Confusion Matrix

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
cm = confusion_matrix(y_pred, y_test,)
print(cm)
print(accuracy_score(y_pred,y_test))