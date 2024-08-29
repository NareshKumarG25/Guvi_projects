#script to get state list along with routes and buses 
#using selenium scapping from redbus website
import red_bus_selenium
import time

def main():
    start = time.time()
    red_bus_selenium.red_bus_scrapping()
    print("Program processing time = :", (time.time()-start)/60)
    print("------END-------")
    
if __name__ == '__main__':
    main()