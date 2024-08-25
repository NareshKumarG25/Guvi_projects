import sqlite3

class GUVI_DB():

    def __init__(self,db_name):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.create_table()
        print("created DB : ",db_name)

    def create_table(self):
        self.cursor.execute ('CREATE TABLE IF NOT EXISTS state_names(\
                state_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,\
                 state_name VARCHAR(100) NOT NULL UNIQUE )')
        
        self.cursor.execute('CREATE TABLE IF NOT EXISTS bus_routes (\
               state_id INTEGER NOT NULL,\
               route_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,\
               route_name VARCHAR(100) NOT NULL ,\
               route_link VARCHAR(100) NOT NULL,\
               FOREIGN KEY (state_id) REFERENCES state_names (state_id)\
               )') 
        
        self.cursor.execute('CREATE TABLE IF NOT EXISTS buses (\
               bus_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,\
               route_id INTERGER NOT NULL,\
               bus_name VARCHAR(75) NOT NULL,\
               bus_type VARCHAR(75) NOT NULL,\
               departing_time TIME,\
               duration TEXT,\
               reaching_time TIME,\
               star_rating FLOAT,\
               price DECIMAL,\
               seats_available INTEGER,\
               FOREIGN KEY (route_id) REFERENCES bus_routes (route_id)\
               )')
        
    def state_db_update(self,parameter):
            self.cursor.execute('INSERT INTO state_names (state_id,state_name) values (?,?)',parameter)

    def route_db_update(self,parameter):
          self.cursor.execute('INSERT INTO bus_routes (state_id,route_id,route_name,route_link) values (?,?,?,?)',parameter)

    def bus_db_update(self,parameter):
          self.cursor.execute('INSERT INTO buses (route_id,bus_name,bus_type,\
                              departing_time,duration,reaching_time,\
                              star_rating,price,seats_available) values (?,?,?,?,?,?,?,?,?)',parameter)

    def commit_the_changes(self):
            self.conn.commit()
