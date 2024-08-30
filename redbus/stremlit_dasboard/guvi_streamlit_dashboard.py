#Streamlit application file to show all bus routes and buses \
#fetched from red bus website
import streamlit as st
import sqlite3
import time
import pandas as pd 
import datetime

place_holder = st.empty()
bus_count_place = st.empty()
table_holder = st.empty()

main_radio=st.sidebar.radio("Choose to move",['Home',"State"])
st.logo("redbus/stremlit_dasboard/guvi_logo.png",link="https://www.guvi.in")
st.sidebar.markdown(
    """
    <div style="position: fixed; bottom:0;height:10%; width:25%; text-align: center;">
        <a href="https://www.guvi.in" target="_blank">Learn from Guvi 💡</a>
    </div>
    """,
    unsafe_allow_html=True
    )

if main_radio =="Home":
    st.image("redbus/stremlit_dasboard/guvi_logo.png", width=150)
    st.header("Welcome to Naresh's GUVI Dashboard!!!!")
    st.write("Choose state in the side bar to explore buses")

    st.markdown(
    """
    <div style="position: fixed; bottom:0;height:25%; width:75%; text-align: center;">
        <a href="https://www.linkedin.com/in/naresh-kumar-8a7a49175" target="_blank">Hire professional data engineer 🎓</a>
    </div>
    """,
    unsafe_allow_html=True
    )

if main_radio == 'State': 
    st.logo("redbus/stremlit_dasboard/redbus.png",link='https://www.redbus.in')
    st.sidebar.title('Red Bus Data')
    try:
        no_value ="Choose an option"
        db_name = 'redbus/database/red_bus_v1.db'
        # db_name= 'D:/Naresh/GUVI/git_testing/Guvi_projects/redbus/database/red_bus.db'
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        cursor.execute("SELECT * from state_names")
        data= cursor.fetchall()

        state_df = pd.DataFrame(data)
        
        state_df.columns=["State ID","State Names"]
        place_holder.text("Showing all States avaialable.")
        table_holder.table(state_df['State Names'])
        
        s_name=st.sidebar.selectbox(
                "Choose state to explore routes",
                state_df["State Names"],
                placeholder = no_value,
                index=None,
                on_change=None,
                )
        
        state_value = 0
        for i in range(len(state_df)):
            if state_df['State Names'][i]==s_name:
                state_value=int(state_df["State ID"][i])
                break

        if s_name != None:
            place_holder.write("Loading ⏳ routes available in {} .....".format(s_name))
            table_holder.empty()
            time.sleep(2)
            
            cursor.execute("SELECT * from bus_routes where state_id=?",(state_value,))
            route_data= cursor.fetchall()

            route_df = pd.DataFrame(route_data)
            if len(route_df):
                place_holder.write("Showing routes available in {} .....".format(s_name))
                bus_count_place.write("Choose any route in side bar to view avaialble buses")
                route_df.columns=["State ID","Route ID","Route Name","Route Link"]
            
                table_holder.table(route_df['Route Name'])
                r_name=st.sidebar.selectbox(
                    "Choose route to explore routes",
                    route_df['Route Name'],
                    placeholder = no_value,
                    index=None,
                    on_change=None,
                    )
                route_value = 0
                for i in range(len(route_df)):
                    if route_df["Route Name"][i]==r_name:
                        route_value=int(route_df["Route ID"][i])
                        break
                if r_name != None:
                    table_holder.empty()
                    place_holder.write("Loading ⏳ buses available in {} .....".format(r_name))

                    cursor.execute("SELECT * from buses where route_id=?",(route_value,))
                    bus_data= cursor.fetchall()
                    time.sleep(2)

                    bus_df = pd.DataFrame(bus_data)

                    if len(bus_df):
                        place_holder.write("Showing buses in route between {}".format(r_name))
                        
                        filters=st.expander("Filters")
                        left, middle, right = filters.columns(3, vertical_alignment="top")
                        dep_time=left.time_input("Departing Time",value=datetime.time(00))
                        reach_time = middle.time_input("Reaching Time ",value=datetime.time(23,59))
                        availablity = right.toggle('Only buses with more than 10 seats',value=True)
                        # filters.button("Reset to default")

                        bus_df.columns=['Bus ID','Route ID','Bus Name','Bus Type','Departing Time',
                                        'Duration','Reaching Time','Rating','Price','Seats available']
                        if availablity:
                            updated_df = bus_df.loc[
                            (bus_df['Seats available']>10) & (bus_df['Departing Time']>=str(dep_time)) &
                                (bus_df['Reaching Time']<=str(reach_time))] 
                        else:
                            updated_df = bus_df.loc[
                            (bus_df['Departing Time']>=str(dep_time)) &
                                (bus_df['Reaching Time']<=str(reach_time))] 
                        bus_count_place.write("{}  buses avalable in route".format(len(updated_df)))
                        table_holder.dataframe(updated_df,hide_index=True)
                        
                    else:
                        bus_count_place.empty()
                        place_holder.write("No buses in route between {}".format(r_name))
            else:
                bus_count_place.empty()
                place_holder.write("No routes available in {} .....".format(s_name))
    finally:
        cursor.close()
        conn.close()
            
