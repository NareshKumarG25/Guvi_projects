import pandas as pd
import ast

def data_processing (file,city,code):
    df = pd.read_excel(file)
    dict_value = df['new_car_detail'].to_dict()

    for i in dict_value:
        actual_value =ast.literal_eval(dict_value[i])
        trending_data = actual_value.pop('trendingText')

        #city_name 
        actual_value['city']=city
        #car_id
        actual_value['car_code']= code+'_'+str(i)
        dict_value[i]=actual_value

        # Add flattened 'trendingText' keys back into the main dictionary
        for key, value in trending_data.items():
            actual_value[f'trending_{key}'] = value
        dict_value[i]=actual_value

        
        # Wrap the dictionary in a list to make it a DataFrame with one row
    df2 = pd.DataFrame.from_dict(dict_value, orient='index')

    dict_value_2 = df['new_car_overview'].to_dict()

    new_dict={}
    for i in dict_value_2:
        actual_value = ast.literal_eval(dict_value_2[i])
        out_dict={}
        for j in actual_value['top']:
            out_dict[j['key']]=j['value']
        new_dict[i]=out_dict

    df3 = pd.DataFrame.from_dict(new_dict, orient='index')


    new_dataframe =df2.join(df3, how='outer')

    dict_value_3 = df['new_car_specs'].to_dict()
    new_dict={}
    for i in dict_value_3:
        actual_value = ast.literal_eval(dict_value_3[i])
        out_dict={}
        for k in actual_value['data']:
            for j in k['list']:
                out_dict[j['key']]=j['value']

        for j in actual_value['top']:
            out_dict[j['key']]=j['value']
        
        new_dict[i]=out_dict

    df4 = pd.DataFrame.from_dict(new_dict, orient='index')

    new_dataframe =new_dataframe.join(df4, how='outer',lsuffix='_df1', rsuffix='_df2')

    dict_value_4 = df['new_car_feature'].to_dict()

    new_dict={}
    for i in dict_value_4:
        actual_value = ast.literal_eval(dict_value_4[i])
        out =[]

        for k in actual_value['data']:
            for j in k['list']:
                out.append(j["value"])
        for j in actual_value['top']:
            out.append(j['value'])
        new_dict[i]=str(out)
    df5 = pd.DataFrame.from_dict(new_dict, orient='index',columns=['features'])
    new_dataframe =new_dataframe.join(df5, how='outer',lsuffix='_df1', rsuffix='_df2')

    return new_dataframe