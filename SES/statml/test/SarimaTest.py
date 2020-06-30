import json
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from statml.modules import SarimaModel as sarima
import os
script_path = os.path.abspath(__file__)
path_list = script_path.split(os.sep)
script_directory = path_list[0:len(path_list) - 2]


def read_input_file(file_path):
    if file_path.endswith(".json"):
        inputparams = json.load(open(file_path))
        algo_info = inputparams['algorithm'][0]
        p = algo_info['p']
        d = algo_info['d']
        q = algo_info['q']
        P = algo_info['P']
        D = algo_info['D']
        Q = algo_info['Q']
        period = algo_info['period']
        trainSet = algo_info['trainSet']
        testSet = algo_info['testSet']
        forecastSteps = algo_info['forecastSteps']


        data_info = {}
        data_info["p"] = p
        data_info["d"] = d
        data_info["q"] = q
        data_info["P"] = P
        data_info["D"] = D
        data_info["Q"] = Q
        data_info["period"] = period
        data_info["trainSet"] = trainSet
        data_info["testSet"] = testSet
        data_info["foreCastSteps"] = forecastSteps
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
        data_info["p"] = 2
        data_info["d"] = 1
        data_info["q"] = 2
        data_info["P"] = 1
        data_info["D"] = 0
        data_info["Q"] = 1
        data_info["period"] = 12
        data_info["trainSet"] = 70
        data_info["testSet"] = 30
        data_info["foreCastSteps"] = 12


        return dataframe, data_info
    else: raise ValueError("The file is not json nor csv")


#rel_path = "dataset/candy_production.csv"

rel_path = "dataset/SeasonalARIMASampleReq.json"
dataset_path = "/".join(script_directory) + "/" + rel_path

df, data_info = read_input_file(dataset_path)
print("df:::::::::::", df)
#print("data_info:::::::::::::::::", data_info)
trainSet = data_info['trainSet']
p = data_info['p']
d = data_info['d']
q = data_info['q']
P = data_info['P']
D = data_info['D']
Q = data_info['Q']
period = data_info['period']


foreCastSteps = data_info['foreCastSteps']
model = sarima.sarima_model(df,p,d,q,P,D,Q,foreCastSteps)
#print("::::::::::::::model",model)

result = sarima.model_fit(model)
#print("::::::::result",result)
result_summary = sarima.summary(result)
#print(":::::::::::::summary",result_summary)

forecast_date = "2013-05"
df_forecast = sarima.forecast(result,forecast_date,df)
#print("::::::::df_forecast",df_forecast)

pred_ci = sarima.forecast_range(result,foreCastSteps)
#print("::::::::::::::pred_ci",pred_ci)

eval_result = sarima.evaluate_model(df, forecast_date, df_forecast)
#print("::::::::::::::eval_result",eval_result)














