
import pandas as pd
import numpy as np
import statsmodels.api as sm



def sarima_model(y,p,d,q,P,D,Q,foreCastSteps):
    model = sm.tsa.statespace.SARIMAX(y,
                                      order=(p,d,q),
                                      seasonal_order=(P,D,Q,foreCastSteps),
                                      enforce_stationarity=False,
                                      enforce_invertibility=False)
    return model
def model_fit(model):
    result = model.fit(disp=False)
    return result

def summary(result):
    return result.summary()

def forecast(result,forecast_date,df):
    forecast = result.get_prediction(start=pd.to_datetime(forecast_date),dynamic=False)
    cols = df.columns
    df_forecast = forecast.predicted_mean.to_frame(name=cols[0])
    return df_forecast

def forecast_range(result,foreCastSteps):
    pred_uc = result.get_forecast(steps=foreCastSteps)
    pred_ci = pred_uc.conf_int()
    return pred_ci

def evaluate_model(y, forecast_date, df_forecast):
    dict={}
    column = y.columns
    actual = y[forecast_date:]
    mse = round(((df_forecast - actual) ** 2).mean(), 2)
    rmse = np.sqrt(mse)
    column_dict={}
    column_dict["mse:"] = mse
    column_dict["rmse:"] = round(rmse, 2)
    dict[column[0]] = column_dict
    return dict


























