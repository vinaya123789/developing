import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

def exponential_smoothing(y):
    model = SimpleExpSmoothing(y)
    return model

#fit the model
def model_fit(model):
    result = model.fit()
    return result
    print(result)

#model summary
def summary(result):
    return result.summary()

#predict the next values
def forecast(result,forecast_date):
    df_forecast = result.predict(start=forecast_date.index("0"),end=forecast_date.index("10"))
    return df_forecast








