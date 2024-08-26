from scraping_guvi import GUVI_SCRAPING
from db_update_guvi import GUVI_DB
import time

website = "https://www.redbus.in/"
db_path='D:/Naresh/GUVI/Redbus/database/'
db_name ='red_bus.db'
db_to_store=db_path+db_name

#to update the database with fetched values
def table_creation(obj_redbus):
    db_object =GUVI_DB(db_to_store)
    state_id =0
    route_id = 0
    for states in obj_redbus.state_dict.keys():
        state_id+=1
        print(states)
        #to add entry in state_name db 
        db_object.state_db_update((state_id,states,))
        if obj_redbus.state_dict[states]:
            for routes in obj_redbus.state_dict[states].keys():
                route_id+=1
                #to add entry in routes db along with state_name reference and route link
                db_object.route_db_update((state_id,route_id,routes,obj_redbus.state_dict[states][routes]['href'],))
                for bus in obj_redbus.state_dict[states][routes]['buses']:
                    #to add entry in bus db along with route id reference
                    db_object.bus_db_update(tuple([route_id]+bus))
    db_object.commit_the_changes()

#to execute scraping class 
def red_bus_scrapping():
    
    obj_redbus = GUVI_SCRAPING(website)
    obj_redbus.explore_page('goverment_bus')
    obj_redbus.get_states()
    time.sleep(10)
    table_creation(obj_redbus)

    