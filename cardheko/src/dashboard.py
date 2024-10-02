import streamlit as st

st.write(" GUVI Dashboard for Used car price prediction")

st.selectbox("Select city",['Chennai',"Bangalore"])

st.text_input(label="Choose Car brand")

st.text_input(label="Variant")

st.text_input(label="Registered Year")

button= st.button("Predict Price")

if button:
    st.write("Predicted Price : YTD")
    st.write("Model Training is in progress")