import joblib
import pandas as pd
from common_functions import convert_number_to_currency

# Step 9: Function to load model and predict based on user input
def predict_price(user_input):
    # Load the saved model
    path =r'D:\Naresh\GUVI\Projects\CarDheko\data\processed_data\RandomForestRegressor_used_car_price_model.pkl'
    model = joblib.load(path)
    
    # Convert the user input into a DataFrame (assuming user_input is a dictionary)
    user_df = pd.DataFrame([user_input])
    
    # Predict the price
    predicted_price = model.predict(user_df)
    
    return predicted_price[0]

# Example usage of predict_price function
user_input = {
    'city': 'Bangalore',
    'manufacturer': 'Maruti',
    'model': 'Maruti Celerio',
    'variant_type': 'VXI',
    'fuel_type': 'Petrol',
    'model_year': 2015,
    'number_of_owners': 3,
    'km_driven': 120000,
    'year_of_registration': 2015
}

predicted_price = predict_price(user_input)

print(f"Predicted price for the car: {convert_number_to_currency(predicted_price)}")
