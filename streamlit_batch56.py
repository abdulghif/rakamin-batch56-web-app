import streamlit as st

st.title('Churn Prediction')
st.write("Most objects") 
st.sidebar.title('Navigasi')

nama = st.text_input('Masukkan Namamu:')

if st.button('Submit'):
    st.write(f'Hello! Nama saya {nama}')