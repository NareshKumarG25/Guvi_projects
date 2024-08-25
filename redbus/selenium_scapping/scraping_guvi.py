import time
from selenium import webdriver
from selenium.webdriver.common.by import By

class GUVI_SCRAPING:
    #xpaths
    goverment_bus = '//a[@href="https://www.redbus.in/online-booking/rtc-directory"]'
    state_route ='//a[@class="D113_link"]'
    bus_route='//a[@class="route"]'

    state_dict ={}
    page_count=0

    def __init__(self,website):
        self.driver = webdriver.Chrome()
        self.driver.get(website)
        time.sleep(5)

    def new_tab(self,action,new_url=''):
        if action == "open":
            self.driver.execute_script("window.open('');")
            self.page_count+=1
            self.driver.switch_to.window(self.driver.window_handles[self.page_count])
            self.driver.get(new_url)
            time.sleep(3)
            self.lazy_loading_scroll()
            return (True,"new page opened and loaded fully")
        else:
            self.driver.close()
            time.sleep(3)
            self.page_count-=1
            self.driver.switch_to.window(self.driver.window_handles[self.page_count])

    def lazy_loading_scroll(self):
        previous_height=self.driver.execute_script('return document.body.scrollHeight')
        while True:
            self.driver.execute_script('window.scrollTo(0,document.body.scrollHeight)')
            time.sleep(3)
            new_height = self.driver.execute_script('return document.body.scrollHeight')
            if new_height==previous_height:
                time.sleep(3)
                break
            previous_height=new_height
    
    def explore_page(self,parameter):
        if parameter == 'goverment_bus':
            new_link=self.driver.find_element(By.XPATH,self.goverment_bus).get_attribute('href')
            self.new_tab('open',new_link)

    def get_bus_details(self,route_link):

        bus_detail_dict={}
        bus_detail_dict['href']=route_link
        self.new_tab('open',route_link)

        # write code to click view buses in grouped tab 

        buses_list= self.driver.find_elements(By.XPATH,'//div[@scrollthreshold="1"]')
        bus_list = []
        if buses_list:
            buses= buses_list[0].find_elements(By.XPATH,'//div[@class="clearfix row-one"]')

            for bus in buses:
                bus_name = bus.find_element(By.XPATH,'.//div[@class="travels lh-24 f-bold d-color"]').text
                bus_type = bus.find_element(By.XPATH,'.//div[@class="bus-type f-12 m-top-16 l-color evBus"]').text 
                departing_time = bus.find_element(By.XPATH,'.//div[@class="dp-time f-19 d-color f-bold"]').text 
                duration = bus.find_element(By.XPATH,'.//div[@class="dur l-color lh-24"]').text 
                reaching_time= bus.find_element(By.XPATH,'.//div[@class="bp-time f-19 d-color disp-Inline"]').text
                try:
                    star_rating = bus.find_element(By.XPATH,'.//div[@class="rating-sec lh-24"]').text
                except:
                    try:
                        star_rating = bus.find_element(By.XPATH,'.//span[@class="ppl_badge ppl_wrap rate_count bo_tag_text"]').text 
                    except:
                        star_rating="No Rating"
                try:
                    price = bus.find_element(By.XPATH,'.//span[@class="f-19 f-bold"]').text 
                except:
                    price = bus.find_element(By.XPATH,'.//span[@class="f-bold f-19"]').text 
                try:
                    seat_availability = bus.find_element(By.XPATH,'.//div[@class="seat-left m-top-30"]').text.split()[0]
                except:
                    seat_availability = bus.find_element(By.XPATH,'.//div[@class="seat-left m-top-16"]').text.split()[0]
                bus_list.append([bus_name,bus_type,departing_time,duration,reaching_time,star_rating,price,seat_availability])

        bus_detail_dict['buses']=bus_list
        self.new_tab('close')
        return bus_detail_dict

    def get_bus_routes(self,state_link):
        self.new_tab('open',state_link)

        #To enable multiple page moving option from lazy loading in main window
        try:
            js_code = "arguments[0].scrollIntoView();"
            element = self.driver.find_element(By.CLASS_NAME, "DC_117_paginationTable")
            self.driver.execute_script(js_code, element)
            time.sleep(2)
        except:
            #to handle routes with no routes avaialble 
            pass

        route_page = 1
        route_dict={}
        while True:
            route_details= self.driver.find_elements(By.XPATH,self.bus_route)
            for routs in route_details:
                bus_details=self.get_bus_details(routs.get_attribute('href'))
                route_dict[routs.text]=bus_details
            try:
                #to handle multiple pages in route page
                next_page = self.driver.find_elements(By.XPATH,'//div[@class="DC_117_pageTabs "]')
                next_page[route_page-1].click()
                print("processed page :",route_page)
                route_page+=1
            except:
                print("End of Page")
                break
        self.new_tab('close')
        return route_dict
    
    def get_states(self):
        state_list = self.driver.find_elements(By.XPATH,self.state_route)
        for states in state_list:
            print("processing state :",states.text)
            route_details=self.get_bus_routes(states.get_attribute("href"))
            self.state_dict[states.text] = route_details
