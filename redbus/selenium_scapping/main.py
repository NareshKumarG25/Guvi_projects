import red_bus_selenium
import time

def main():
    start = time.time()
    red_bus_selenium.red_bus_scrapping()
    print("Program processing time = :", (time.time()-start)/60)
    
if __name__ == '__main__':
    main()