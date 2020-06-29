from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np


def exponential_smoothing(trainSet):
    model = ExponentialSmoothing(trainSet, damped=False, seasonal="add", seasonal_periods=None, dates=None,
                                 missing='none')

    return model


# fit the model
def model_fit(model):
    result = model.fit()

    return result


# model summary
def summary(result):
    return result.summary()


def forecast(result, testSet):
    df_forecast = result.predict(start=testSet.index[0], end=testSet.index[-1])
    return df_forecast


# predict the next values
def prediction(testSet, forecastSteps):
    df_predict = testSet.forecast(forecastSteps)
    return df_predict


# calculate rmse and mse value w.r.t columns
def evaluate_model(df_predict):
    dict = {}
    column = df_predict.columns
    mse = round(((df_predict) ** 2).mean(), 2)
    rmse = np.sqrt(mse)
    column_dict = {}
    column_dict["mse:"] = mse
    column_dict["rmse:"] = round(rmse, 2)
    dict[column[0]] = column_dict
    return dict
