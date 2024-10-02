import pandas as pd
from raw_data_processing import data_processing

if __name__ == '__main__':
    
    over_all_dataframe = pd.DataFrame()

    path = "D:/Naresh/GUVI/Projects/CarDheko/data/raw/"

    list_of_data = {'bangalore':{'file':'bangalore_cars.xlsx','code':'blr'},
                    'chennai':{'file':'chennai_cars.xlsx','code':'chn'},
                    'delhi':{'file':'delhi_cars.xlsx','code':'dhl'},
                    'hydrabad':{'file':'hyderabad_cars.xlsx','code':'hyd'},
                    'jaipur':{'file':'jaipur_cars.xlsx','code':'jap'},
                    'kolkata':{'file':'kolkata_cars.xlsx','code':'kol'}
                    }

    for i in list_of_data:
        print("Processing city : ",i)
        new_dataframe=data_processing(path+list_of_data[i]['file'],i,list_of_data[i]['code'])

        over_all_dataframe=pd.concat([over_all_dataframe,new_dataframe],ignore_index=True)

    print("Writing in Excel.....")
    with pd.ExcelWriter('raw_processed.xlsx') as writer:
        over_all_dataframe.to_excel(writer, sheet_name='raw_processed',index=False)