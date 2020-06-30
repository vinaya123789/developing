import json
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
import os

script_path = os.path.abspath(__file__)
path_list = script_path.split(os.sep)
script_directory = path_list[0:len(path_list) - 2]
from statml.modules import HoltWinterExpoSmoothingCreate as hwesc


def read_input_file(file_path):
    if file_path.endswith(".json"):
        inputparams = json.load(open(file_path))
        algo_info = inputparams['algorithm'][0]
        forecastSteps = algo_info['forecastSteps']

        data_info = {}
        data_info["forecastSteps"] = forecastSteps
        columns = inputparams['data']['columns']
        values = inputparams['data']['values']
        df1 = pd.DataFrame(columns).T.set_index(0)
        df2 = pd.DataFrame(values).set_index(0)
        dataframe = df1.append(df2)
        dataframe = dataframe.rename(columns=dataframe.iloc[0]).drop(dataframe.index[0])
        dataframe.index.name = df1.index[0]
        dataframe = dataframe.apply(pd.to_numeric, errors='coerce')
        return dataframe, data_info

    elif file_path.endswith(".csv"):
        dataframe = pd.read_csv(file_path, parse_dates=True, index_col=0)
        data_info = {}
        data_info["forecastSteps"] = 12

        return dataframe, data_info
    else:
        raise ValueError("The file is not json nor csv")


# rel_path = "dataset/candy_production.csv"
rel_path = "dataset/Holt-winterExponentialSmoothing.json"
dataset_path = "/".join(script_directory) + "/" + rel_path
df, data_info = read_input_file(dataset_path)
# print("::::::::::::::::df", df)
# print("::::::::::::data_info",data_info)
trainSet = df.iloc[:7]
testSet = df.iloc[7:]
model = hwesc.exponential_smoothing(trainSet)
# print(":::::::::::::::::model", model)
result = hwesc.model_fit(model)
# print(":::::::::::::::result", result)
result_summary = hwesc.summary(result)
# print(":::::::::::::::summary", result_summary)

df_forecast = hwesc.forecast(result, testSet)
# print("::::::::::::::::df_forecast",df_forecast)
forecastSteps = data_info['forecastSteps']
df_predict = hwesc.prediction(result, forecastSteps)
# print(":::::::::::df_predict",df_predict)
df_predict = pd.DataFrame(df_predict)
eval_model = hwesc.evaluate_model(df_predict)
# print("::::::::::::::eval_model", eval_model)
