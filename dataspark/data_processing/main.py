import pandas as pd
from dateutil import parser
import numpy as np
from db_update import DATASPARK_DB
import time

file_path = 'D:/Naresh/GUVI/Projects/DataSpark/data/'

db_path='D:/Naresh/GUVI/Projects/DataSpark/database/'
db_name ='data_spark.db'
db_to_store=db_path+db_name

db_object = DATASPARK_DB(db_to_store)

def sales_data():
    count = 1
    print("Reading Sales data...")
    sales=pd.read_csv(file_path+'Sales.csv',encoding='unicode_escape')
    sales = sales.replace(np.nan,None)
    print("Processing sales data...")

    total_count = len(sales)
    for sale in range(total_count):
        print("\rprocessed {}/{}".format(count,total_count),end='')
        sales.loc[sale,'Order Date'] = parser.parse(sales['Order Date'][sale]).strftime("%Y-%m-%d")

        if sales.loc[sale,'Delivery Date'] != None:
            sales.loc[sale,'Delivery Date'] = parser.parse(sales['Delivery Date'][sale]).strftime("%Y-%m-%d")

        parameter = tuple(sales.values[sale])
        db_object.sales_table_update(parameter)
        count+=1
    print("\nSales data completed !")

def customer_data():
    count = 1
    print("Reading customer data...")
    customer=pd.read_csv(file_path+'Customers.csv',encoding='unicode_escape')
    customer = customer.replace(np.nan,None)
    print("Processing customer data...")

    total_count = len(customer)

    for data in range(total_count):
        print("\rprocessed {}/{}".format(count,total_count),end='')

        customer.loc[data,'Birthday'] = parser.parse(customer['Birthday'][data]).strftime("%Y-%m-%d")
        customer.loc[data,'Gender'] = customer.loc[data,'Gender'].upper()
        parameter = tuple(customer.values[data])
        db_object.customer_table_update(parameter)
        count+=1

    print("\ncustomer data completed !")

def exchange_rate():

    count = 1
    print("Reading exchange rate data...")
    exchange=pd.read_csv(file_path+'Exchange_Rates.csv',encoding='unicode_escape')
    exchange = exchange.replace(np.nan,None)
    print("Processing customer data...")

    total_count = len(exchange)

    for data in range(total_count):
        print("\rprocessed {}/{}".format(count,total_count),end='')

        exchange.loc[data,'Date'] = parser.parse(exchange['Date'][data]).strftime("%Y-%m-%d")
        parameter = tuple(exchange.values[data])
        db_object.exchange_table_update(parameter)
        count+=1

    print("\nExchange data completed !")

def product_data():
    count = 1
    print("Reading product data...")
    products=pd.read_csv(file_path+'Products.csv',encoding='unicode_escape')
    products = products.replace(np.nan,None)
    print("Processing products data...")

    total_count = len(products)

    for product in range(total_count):
        print("\rprocessed {}/{}".format(count,total_count),end='')

        products.loc[product,'Unit Cost USD'] = float(products.loc[product,'Unit Cost USD'].replace("$",'').replace(",",''))
        products.loc[product,'Unit Price USD'] = float(products.loc[product,'Unit Price USD'].replace("$",'').replace(",",''))

        parameter = tuple(products.values[product])
       
        db_object.product_table_update(parameter)
        count+=1

    print("\nProduct data completed !")

def store_data():
    count = 1
    print("Reading store data...")
    stores=pd.read_csv(file_path+'Stores.csv',encoding='unicode_escape')
    stores = stores.replace(np.nan,None)
    print("Processing Store data...")

    total_count = len(stores)

    for store in range(total_count):
        print("\rprocessed {}/{}".format(count,total_count),end='')

        stores.loc[store,'Open Date'] = parser.parse(stores['Open Date'][store]).strftime("%Y-%m-%d")

        parameter = tuple(stores.values[store])
        db_object.store_table_update(parameter)
        count+=1

    print("\nStore data completed !")


def main():
    start_time = time.time()

    sales_data()
    customer_data()
    exchange_rate()
    product_data()
    store_data()
    
    db_object.commit_changes()
    print("Program processing time :", (time.time()-start_time)/60)
    print("------END-------")

if __name__ =='__main__':
    main()
