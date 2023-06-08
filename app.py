import streamlit as st
import pickle

st.title("CalRecycle Data Vizualisation")

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectors.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
    
waste = st.text_input("Enter Waste: ")    
   
text_features = vectorizer.transform([waste])

# Predict the category
category = model.predict(text_features)[0]
st.text(category)
