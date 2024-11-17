import pandas as pd
import raw_data_processing 
import model_data_processing  

if __name__ == '__main__':
    
    over_all_dataframe = pd.DataFrame()

    path = "D:/Naresh/GUVI/Projects/CarDheko/data/raw/"

    output_path = "D:/Naresh/GUVI/Projects/CarDheko/data/processed_data/"

    list_of_data = {'Bangalore':{'file':'bangalore_cars.xlsx','code':'blr'},
                    'Chennai':{'file':'chennai_cars.xlsx','code':'chn'},
                    'Delhi':{'file':'delhi_cars.xlsx','code':'dhl'},
                    'Hydrabad':{'file':'hyderabad_cars.xlsx','code':'hyd'},
                    'Jaipur':{'file':'jaipur_cars.xlsx','code':'jap'},
                    'Kolkata':{'file':'kolkata_cars.xlsx','code':'kol'}
                    }

    for i in list_of_data:
        print("Processing city : ",i)
        new_dataframe=raw_data_processing.data_processing(path+list_of_data[i]['file'],i,list_of_data[i]['code'])

        over_all_dataframe=pd.concat([over_all_dataframe,new_dataframe],ignore_index=True)

    print("Writing in Excel.....")
    with pd.ExcelWriter(output_path+'raw_processed.xlsx') as writer:
        over_all_dataframe.to_excel(writer, sheet_name='raw_processed',index=False)


    model_df = model_data_processing.data_processing(output_path+'raw_processed.xlsx')

    with pd.ExcelWriter(output_path+'model_data.xlsx') as writer:
        print("writing data in excel...")
        model_df.to_excel(writer, sheet_name='raw_processed',index=False)

    model_df_featured = model_data_processing.model_featuring(output_path+'model_data.xlsx')
    with pd.ExcelWriter(output_path+'model_data_featured.xlsx') as writer:
        print("writing data in excel...")
        model_df_featured.to_excel(writer, sheet_name='raw_processed',index=False)