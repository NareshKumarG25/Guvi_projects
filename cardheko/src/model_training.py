# Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib 

#Load dataset
print("Reading Data file...")
file = r'D:\Naresh\GUVI\Projects\CarDheko\data\processed_data\model_data_featured.xlsx'
data = pd.read_excel(file)


#Feature selection (City, Manufacturer, Model, Variant Type, Fuel Type, Model Year, Number of Owners, KM Driven, Year of Registration)
X = data[['city', 'manufacturer', 'model', 'variant_type', 'fuel_type', 'model_year', 'number_of_owners', 'km_driven', 'year_of_registration']]

# Target variable (Price)
y = data['price']

#Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Preprocessing: Encoding categorical variables
categorical_features = ['city', 'manufacturer', 'model', 'variant_type', 'fuel_type']
numeric_features = ['model_year', 'number_of_owners', 'km_driven', 'year_of_registration']

# OneHotEncoder for categorical variables
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ],
    remainder='passthrough'  # Pass through numerical features without transformation
)

#Building a pipeline: Preprocessing + Model
model_name = 'random_forest_regression'
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# model_name = 'linear_regression'
# model_pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('model', LinearRegression())
# ])

# Train the model
print("Training the model...")
model_pipeline.fit(X_train, y_train)

# Save the model to a file
print("Storing the model in file...")
output_path = "D:/Naresh/GUVI/Projects/CarDheko/data/processed_data/"
joblib.dump(model_pipeline, output_path+model_name+'_used_car_price_model.pkl')

#Evaluate model performance (optional)
y_pred_train = model_pipeline.predict(X_train)
y_pred_test = model_pipeline.predict(X_test)


train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)

train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)


print("-------------------------")
print(f"Training RMSE: {train_rmse}")
print(f"Training MAE: {train_mae}")
print(f"R2 scrore Train : {r2_train}")
print("-------------------------")
print(f"Test RMSE: {test_rmse}")
print(f"Test MAE: {test_mae}")
print(f"R2 scrore Test : {r2_test}")
