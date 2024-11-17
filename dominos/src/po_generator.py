import pandas as pd 
from model import predict_quantity

sales_file = r'D:\Naresh\GUVI\Projects\dominos\data\raw\Pizza_Sale.xlsx'
ing_file = r'D:\Naresh\GUVI\Projects\dominos\data\raw\Pizza_ingredients.xlsx'

output_path ='D:/Naresh/GUVI/Projects/dominos/data/processed_file/'

print("Reading Raw files...")
sales_data = pd.read_excel(sales_file)
ing_data = pd.read_excel(ing_file)

print("Handling missing data...")
pizza_id_formater = {}
for name in ing_data['pizza_name'].unique():
    index_of_value = ing_data[ing_data['pizza_name'] == name].index[0]
    value = ing_data['pizza_name_id'][index_of_value]
    last_index =  value.rfind('_')
    value = value[:last_index]
    pizza_id_formater[name]=value

sales_model_df = sales_data[['pizza_name_id','quantity','order_date','pizza_size','pizza_name']]

#for filling missing values
for rows in range(len(sales_model_df)):
    #for pizza ids
    if pd.isna(sales_model_df['pizza_name_id'][rows]):
        pizza_id = pizza_id_formater[sales_model_df['pizza_name'][rows]]+"_"+sales_model_df['pizza_size'][rows].lower()
        # print(f"{pizza_id}: {sales_model_df['pizza_size'][rows]}: {sales_model_df['pizza_name'][rows]}")
        sales_model_df['pizza_name_id'][rows]= pizza_id
    #for pizza name 
    if pd.isna(sales_model_df['pizza_name'][rows]):
        value_to_find= sales_model_df['pizza_name_id'][rows][:sales_model_df['pizza_name_id'][rows].rfind('_')]
        pizza_name = next((key for key, value in pizza_id_formater.items() if value == value_to_find), None)

        # print(f"{value_to_find}: {pizza_name}")
        sales_model_df['pizza_name'][rows]=pizza_name

df_grouped = sales_model_df.groupby(['order_date', 'pizza_name_id'], as_index=False)['quantity'].sum()
df_grouped['order_date'] = pd.to_datetime(df_grouped['order_date'])
df_grouped['day_of_week'] = df_grouped['order_date'].dt.dayofweek 
df_grouped['order_date'] = df_grouped['order_date'].dt.date


file_name = output_path+'model_data_featured.xlsx'
with pd.ExcelWriter(file_name) as writer:
        print("writing data in excel...")
        sales_model_df.to_excel(writer, sheet_name='processed_data',index=False)
        df_grouped.to_excel(writer,sheet_name='grouped_data',index=False)

print(f"Processed file is saved {file_name}")

print("Predicting the quantity...")
modeled_df = pd.read_excel(file_name,sheet_name='grouped_data')

#dataframe to store the predicted values
new_df = pd.DataFrame()
for pizza_id in modeled_df['pizza_name_id'].unique():
    forecast_next_week = predict_quantity(modeled_df, pizza_id, forecast_periods=7)
    # forecast_next_month = predict_quantity(modeled_df, pizza_id, forecast_periods=30)
    new_df = pd.concat([new_df, forecast_next_week], ignore_index=True)
new_df['forecasted_quantity'] = new_df['forecasted_quantity'].round().astype(int)

#grouping the predicted the values based on prizza id 
po_order_df = new_df.groupby(['pizza_name_id'], as_index=False)['forecasted_quantity'].sum()

#pizza wise ingrediant list
ingrediant_list ={}
for id in ing_data['pizza_name_id'].unique():
    ingrediant_list[id]={}

for index, row in ing_data.iterrows():
    ingrediant_list[row['pizza_name_id']][row['pizza_ingredients']]=row['Items_Qty_In_Grams']

po_order_dict={}
for ingre in ing_data['pizza_ingredients'].unique():
    po_order_dict[ingre]=0

print("Calculating quantity of ingrediants....")
for index,rows in po_order_df.iterrows():
    id = rows['pizza_name_id']
    value = rows['forecasted_quantity']
    for ingre in ingrediant_list[id]: 
        po_order_dict[ingre]+= (value*ingrediant_list[id][ingre])

final_df = pd.DataFrame.from_dict(po_order_dict, orient='index')
final_df.reset_index(inplace=True) 
final_df.columns=['Ingrediants','Quantity (in grams)']

output_file_name = output_path+'predicted_output_quantity.xlsx'
with pd.ExcelWriter(output_file_name) as writer:
        print("writing data in excel...")
        final_df.to_excel(writer,sheet_name='Po for ingrediants',index=False)
        po_order_df.to_excel(writer,sheet_name='Pizza Wise',index=False)
        new_df.to_excel(writer,sheet_name='day wise',index=False)
        
print(f"Output is stored in file {output_file_name }")