import joblib
import pandas as pd
from common_functions import convert_number_to_currency
path ='CarDheko/data/processed_data/random_forest_regression_used_car_price_model.pkl'
input_data ='CarDheko/data/processed_data/model_data_featured.xlsx'
class DashboardData():

    def __init__(self):  
        self.trained_model = joblib.load(path)
        self.df= pd.read_excel(input_data)
        self.city_list = list(set(self.df['city']))
        self.car_list =self.get_car_list()

    def get_car_list(self):
        car_list={}
        df=self.df

        for row in range(len(df)):
            if df['manufacturer'][row] not in car_list.keys():
                car_list[df['manufacturer'][row]]={}
            
            if df['model'][row] not in car_list[df['manufacturer'][row]].keys():
                car_list[df['manufacturer'][row]][df['model'][row]]={}

            if df['variant_type'][row] not in car_list[df['manufacturer'][row]][df['model'][row]].keys():
                car_list[df['manufacturer'][row]][df['model'][row]][df['variant_type'][row]]={}
            
            if df['fuel_type'][row] not in car_list[df['manufacturer'][row]][df['model'][row]][df['variant_type'][row]].keys():
                car_list[df['manufacturer'][row]][df['model'][row]][df['variant_type'][row]][df['fuel_type'][row]]={'model_years':[]}

            model_year = int(df['model_year'][row])
            if model_year not in car_list[df['manufacturer'][row]][df['model'][row]][df['variant_type'][row]][df['fuel_type'][row]]['model_years']:
                car_list[df['manufacturer'][row]][df['model'][row]][df['variant_type'][row]][df['fuel_type'][row]]['model_years'].append(model_year)

        return car_list
    
    def predict_price(self,user_input):
        
        # Convert the user input into a DataFrame (assuming user_input is a dictionary)
        user_df = pd.DataFrame([user_input])
        
        # Predict the price
        predicted_price = self.trained_model.predict(user_df)
        
        return convert_number_to_currency(predicted_price[0])