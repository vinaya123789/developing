import json
import pandas as pd
import warnings
import os
from statml.modules import SimpleExponentialSmoothingModel as sesm
script_path = os.path.abspath(__file__)
path_list = script_path.split(os.sep)
script_directory = path_list[0:len(path_list) - 2]

def read_input_file(file_path):
    if file_path.endswith(".json"):
        inputparams = json.load(open(file_path))
        algo_info = inputparams['algorithm'][0]
        trainSize = algo_info['trainSize']
        testSize = algo_info['testSize']
        forecastSteps = algo_info['forecastSteps']


        data_info = {}
        data_info["forecastSteps"] = forecastSteps
        data_info["trainSize"] = trainSize
        data_info["testSize"] = testSize
        columns = inputparams['data']['columns']
        values = inputparams['data']['values']
        df1 = pd.DataFrame(columns).T.set_index(0)
        df2 = pd.DataFrame(values).set_index(0)
        dataframe = df1.append(df2)
        dataframe = dataframe.rename(columns=dataframe.iloc[0]).drop(dataframe.index[0])
        dataframe.index.name = df1.index[0]
        dataframe = dataframe.apply(pd.to_numeric,errors = 'coerce')
        return dataframe,data_info

    elif file_path.endswith(".csv"):
        dataframe = pd.read_csv(file_path,parse_dates=True,index_col=0)
        data_info = {}
        data_info["trainSize"] = 70
        data_info["testSize"]  = 30
        data_info["forecastSteps"] = 12

        return dataframe,data_info
    else: raise ValueError("The file is not json nor csv")

#rel_path = "dataset/candy_production.csv"
rel_path = "dataset/SimpleExponentialSmoothing.json"
dataset_path = "/".join(script_directory) +"/" + rel_path
df,data_info = read_input_file(dataset_path)
#print("::::::::::::::::df", df)
#print("::::::::::::data_info",data_info)


trainSize = data_info['trainSize']
testSize = data_info['testSize']
forecastSteps = data_info['forecastSteps']

model = sesm.exponential_smoothing(df)
#print("::::::::::::model", model)
result = sesm.model_fit(model)
#print("::::::::::::resulkt",result)
result_summary = sesm.summary(result)
#print("::::::::::summary",result_summary)
forecast_date = "2001-01-10"
df_forecast = sesm.forecast(result,forecast_date)
print("::::::::::::df_forecast",df_forecast)









