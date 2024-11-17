import streamlit as st
import time
from dashboard_data import DashboardData

data_obj = DashboardData()


st.write("GUVI Dashboard for Used car price prediction")
cola, colb = st.columns(2)
with cola:
    city= st.selectbox("City",data_obj.city_list,index=None,placeholder="Choose an City")
with colb:
    manufacturer=st.selectbox('Brand',data_obj.car_list.keys(),index=None,placeholder="Choose an Brand")

if manufacturer and city:
    model=st.selectbox("Model",data_obj.car_list[manufacturer].keys(),index=None,placeholder="Choose an Model")
    if model:
        with st.popover("Click to Change variant"):
            st.markdown("Choose Variant Type")
            variant_type=st.selectbox("Choose Car Variant",data_obj.car_list[manufacturer][model].keys())
            fuel_type=st.radio("Choose Car Fuel Type",data_obj.car_list[manufacturer][model][variant_type].keys())
            model_year =st.selectbox("Choose car Model Year",data_obj.car_list[manufacturer][model][variant_type][fuel_type]['model_years'])
        left, middle, right = st.columns(3)
        with left:
            st.markdown(f"<p style='color: green;'>Variant Type : {variant_type}</p>", unsafe_allow_html=True)
        with middle:
            st.markdown(f"<p style='color: red;'>Fuel Type : {fuel_type}</p>", unsafe_allow_html=True)
        with right:
            st.markdown(f"<p style='color: blue;'>Model Year : {model_year}</p>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            number_of_owners=st.number_input(label="No of Owners",min_value=1,format="%d")
        with col2:
            km_driven=st.number_input(label="kM Driven",min_value=1000,max_value=250000,step=1000,format="%d")
        with col3:
            year_of_registration=st.selectbox("Year Of Registeration",list(range(model_year,2024)),index=None,placeholder="Choose Registred Year")


        user_input = {
            'city': city,
            'manufacturer': manufacturer,
            'model': model,
            'variant_type': variant_type,
            'fuel_type': fuel_type,
            'model_year': model_year,
            'number_of_owners': number_of_owners,
            'km_driven': km_driven,
            'year_of_registration': year_of_registration
        }

        if year_of_registration:
            button= st.button("Predict Price")
        else:
            button= st.button("Predict Price",disabled=True)

        if button:
            price=data_obj.predict_price(user_input)

            progress_text = "Price Prediction in progress. Please wait."
            my_bar = st.progress(0, text=progress_text)

            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1, text=progress_text)
            time.sleep(1)
            my_bar.empty()
            st.write(f"Predicted Price : {price}")
            # st.write("Note : Value may vary with actual. Model Improvement in progress")