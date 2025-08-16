import streamlit as st

import pandas as pd

import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.title('Iris Prediction')

st.header('Modeling Section')
if st.button('Lakukan Modeling'):
    ## Baca Data Iris
    st.write('Membaca Data ...')
    df = pd.read_csv('data/iris.csv')
    
    st.write('Data sudah terbaca...')

    X = df.drop(['Id','Species'],axis=1)
    y = df['Species']
    
    st.write('Process memisahkan training dan testing...')
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)
    rfc = RandomForestClassifier()

    st.write('Proses training model...')
    rfc.fit(X_train,y_train)

    accuracy = rfc.score(X_test,y_test)
    st.write(f'Accuracy: {accuracy}')
    
    st.write('Modeling selesai dilatih maka berikutnya melakukan penyimpanan model')
    
    path_model = 'model/iris.pkl'
    with open(path_model, 'wb') as file:
        pickle.dump(rfc,file)
    
    st.write(f'Model telah disimpan di {path_model}')
    
    
# Section Prediction
st.header('Prediction Iris')

st.write('Jika kamu sudah melakukan modeling, maka kamu dapat memprediksi kelas Iris dengan variable yang sudah dilatih')

st.header('Masukkan input variable')
petal_length = st.number_input("Petal Length")
petal_width = st.number_input("Petal Width")
sepal_length = st.number_input("Sepal Length")
sepal_width = st.number_input("Sepal Width")

X_pred = [[sepal_length,sepal_width,petal_length,petal_width]]

if st.button('Predict!'):
    
    
    with open('model/iris.pkl', 'rb') as file:
        rfc = pickle.load(file)
        
    y_pred = rfc.predict(X_pred)   
    st.write(f'Prediction: {y_pred[0]}')