import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()
df = pd.DataFrame(breast_cancer_dataset.data,columns=breast_cancer_dataset.feature_names)
print(df)
print(df.columns)
print(df.info())
print(df.describe())
df['target'] = breast_cancer_dataset.target
print(df.tail())
print(df.isnull().sum())
print(df.groupby('target').mean())

y = df['target']
X = df.drop('target',axis=1)

print(X)
print(y)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.fit_transform(X_test)
import tensorflow as tf
tf.random.set_seed(3)
import keras
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (30,)),
    keras.layers.Dense(30,activation = 'relu'),
    keras.layers.Dense(30,activation = 'relu'),
    keras.layers.Dense(2, activation='sigmoid')
])
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(X_train_std,y_train,epochs=20)

loss,accuracy = model.evaluate(X_test_std,y_test)
print(accuracy)

#predictor
input_data = (11.76,21.6,74.72,427.9,0.08637,0.04966,0.01657,0.01115,0.1495,0.05888,0.4062,1.21,2.635,28.47,0.005857,0.009758,0.01168,0.007445,0.02406,0.001769,12.98,25.72,82.98,516.5,0.1085,0.08615,0.05523,0.03715,0.2433,0.06563)
input_data_as_np = np.asarray(input_data)
input_data_reshaped= input_data_as_np.reshape(1,-1)
input_data_std = scaler.transform(input_data_reshaped) 
prediction = model.predict(input_data_std)
prediction_label = [np.argmax(prediction)]
if(prediction_label[0]==0):
    print('Malignant')
else:
    print('Benign')