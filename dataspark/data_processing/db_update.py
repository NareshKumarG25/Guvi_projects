import sqlite3

class DATASPARK_DB():

    def __init__(self,db_name):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.create_table()
        print("created DB : ",db_name)

    def create_table(self):
        self.cursor.execute ('CREATE TABLE IF NOT EXISTS sales(\
                order_number INTEGER NOT NULL,\
				line_item INTEGER NOT NULL,\
				order_date DATE,\
				delivery_date DATE,\
				customer_key INTEGER,\
				store_key INTEGER,\
				product_key INTEGER,\
				quantity INTEGER,\
				currency_code VARCHAR(10) )')
        
        self.cursor.execute ('CREATE TABLE IF NOT EXISTS customer(\
                customer_key INTEGER NOT NULL PRIMARY KEY,\
				gender VARCHAR(10) NOT NULL,\
				name VARCHAR(50),\
				city VARCHAR(25),\
				state_code VARCHAR(5),\
				state VARCHAR(20),\
				zip_code INTEGER,\
				country VARCHAR(10),\
				continent VARCHAR(10),\
                birthday DATE )')
        
        self.cursor.execute ('CREATE TABLE IF NOT EXISTS exchange(\
                date DATE NOT NULL,\
				currency VARCHAR(10) NOT NULL,\
				exchange FLOAT )')
        
        self.cursor.execute('CREATE TABLE IF NOT EXISTS products(\
                            product_key INTEGER NOT NULL PRIMARY KEY,\
                            product_name VARCHAR(100) NOT NULL,\
                            brand VARCHAR(30),\
                            color VARCHAR (30),\
                            unit_cost_usd FLOAT ,\
                            unit_price_usd FLOAT ,\
                            sub_catergory_key INTEGER,\
                            sub_category VARCHAR(30),\
                            category_key INTEGER,\
                            category VARCHAR(30))')
        
        self.cursor.execute ('CREATE TABLE IF NOT EXISTS stores(\
                store_key INTEGER NOT NULL PRIMARY KEY,\
				country VARCHAR(25) NOT NULL,\
                state VARCHAR(25),\
                square_meter INTEGER,\
				opening_date DATE )')
        
    def commit_changes(self):
        self.conn.commit()

    def sales_table_update(self,parameter):
        self.cursor.execute('INSERT INTO sales (order_number,line_item,\
                            order_date,delivery_date,customer_key,store_key,\
                           product_key,quantity,currency_code ) \
                            values (?,?,?,?,?,?,?,?,?)',parameter)
        
    def customer_table_update(self,parameter):
        self.cursor.execute('INSERT INTO customer (customer_key,gender,\
                            name,city,state_code,state,zip_code,country,\
                            Continent,birthday ) \
                            values(?,?,?,?,?,?,?,?,?,?) ', parameter)
        
    def exchange_table_update(self,parameter):
        self.cursor.execute('INSERT INTO exchange (date,currency,exchange)\
                            values(?,?,?)',parameter)
        
    def product_table_update(self,parameter):
        self.cursor.execute('INSERT INTO products (product_key,product_name,brand,\
							color,unit_cost_usd,unit_price_usd,\
							sub_catergory_key,sub_category,\
							category_key,category)\
                            values (?,?,?,?,?,?,?,?,?,?)',parameter)

    def store_table_update(self,parameter):
        self.cursor.execute('INSERT INTO stores (store_key,country,state,\
                            square_meter,opening_date)\
                            values(?,?,?,?,?)',parameter)


    