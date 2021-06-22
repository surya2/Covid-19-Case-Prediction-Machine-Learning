#Installations
import os
import os.path
import tensorflow as tf

# Basic Data Science Frameworks
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import statistics

# Sklearn Libraries
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# Statsmodels (Arima model integration) libraries
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import register_matplotlib_converters
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
register_matplotlib_converters()

# Scipy libraries (Piece-wise Interp)
from scipy.interpolate import splev, splrep
from scipy.optimize import curve_fit

# Utilities
from datetime import datetime
import datetime as dt
import math
import warnings
warnings.filterwarnings("ignore")
# multivariate multi-step encoder-decoder lstm
import collections
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot

import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, RNN, Bidirectional, LSTMCell
from keras.layers import Flatten
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard


# Code to define all of the smoothing and live fitting functions
def get_derivatives(cases):
    num_days_in_pandemic = [*range(0, len(cases), 1)]
    instaneous_points = []
    for i in range(1, len(num_days_in_pandemic)):
        instaneous_points += np.linspace(i - 1, i, 1000).tolist()

    spl = splrep(num_days_in_pandemic, cases)
    instantaneous_cases = splev(instaneous_points, spl)

    derivatives = []
    for i in range(0, len(cases)):
        if i == 0:
            derivatives.append(0)
        else:
            number_of_obs = i * 1000
            rate_of_infection = cases[i] - instantaneous_cases[number_of_obs - 1]
            derivatives.append(rate_of_infection)
    return derivatives


def generate_scale(n_past, training_data_len, dataset):
    # Scale and Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    print(dataset.shape)
    print(scaled_data.shape)

    # Create training dataset & Create the scaled training dataset
    train_data = scaled_data[0:training_data_len, :]
    # print(train_data.shape)

    # Split the data into x_train and y_train data sets
    x_train, y_train = [], []
    for i in range(n_past, len(train_data)):
        x_train.append(train_data[i - n_past:i, :])
        y_train.append(train_data[i, 0])

    # print(x_train)
    return x_train, y_train, scaler, scaled_data


def exponential(x, a, b):
    return a * np.exp(b * x)


def piecewise_smoothing(date, day, y):
    x_smooth = np.linspace(min(date), max(date), 1000)
    x_smoothTime = np.linspace(min(day), max(day), 1000)
    data = {'Date': x_smooth,
            'time': x_smoothTime
            }
    smoothX = pd.DataFrame(data, columns=['Date', 'time'])
    spl = splrep(date, y)
    y_smooth = splev(x_smooth, spl)
    # y_smoothTime = splev(x_smoothTime, spl)
    return x_smooth, y_smooth, smoothX, x_smoothTime


def movAvg(values, window):
    weights = np.repeat(1.0, window) / window
    smas = np.convolve(values, weights, 'valid')
    return smas
    # Use a window of 7 days or lower


def expMovAvg(values, window):
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    # May have to take out

    emas = np.convolve(values, weights)[:len(values)]
    emas[:window] = emas[window]
    return emas


# Shallow-Model Case Predictor - For Optimization
class shallow_opModel:
    def __init__(self, state, cases, deaths, rates, stateDates, pplFullVaccinated):
        self.state = state

        '''Pull Github Data'''
        self.dailyCases = cases
        self.dailyDeaths = deaths
        self.ratesOfInfection = rates
        self.pplFullyVaccinated = pplFullVaccinated
        self.stateDates = stateDates

        '''Get Statistical Metrics'''
        self.mean = statistics.mean(self.dailyCases)
        self.variance = statistics.variance(self.dailyCases, xbar=self.mean)
        self.std = statistics.stdev(self.dailyCases, xbar=self.mean)

        # print(len(self.dailyCases.get(state)))
        '''Create slice dataframe for all data to be used'''
        self.data = {"date": self.stateDates,
                     "deaths": self.dailyCases,
                     "cases": self.dailyCases,
                     "rates": self.ratesOfInfection,
                     "pplFullyVaccinated": self.pplFullyVaccinated}

        '''if(len(cases) > 1 and len(rates) > 1):
            self.data = {"date": self.stateDates.get(state),
                "deaths": self.dailyCases.get(state), 
                "cases": cases,
                "rates": rates,
                "pplFullyVaccinated": self.pplFullyVaccinated}'''

        self.batch_sizeList, self.neuronCountList, self.learning_rateList, self.dropoutList, self.rmseList, self.pctErrorList = [], [], [], [], [], []  # Initialize all search spaces
        self.df = pd.DataFrame(self.data, columns=["date", "cases", "rates"])
        self.df.set_index('date', inplace=True)

        '''Filter out features and store in datasets'''
        self.dataFromDF = self.df.filter(["cases", "rates"])
        self.datasetForTheLastTime = self.dataFromDF.values
        self.dataset = self.dataFromDF.values

        '''Specify Training Data length and number of observations to look at '''
        self.training_data_len = math.ceil(len(self.dataset) * .8)
        self.n_past = 30

        print(len(self.dataset))
        self.x_train, self.y_train, self.scaler, self.scaled_data = generate_scale(self.n_past,
                                                                                        self.training_data_len,
                                                                                        self.dataset)  # Scale and Normalize data
        self.x_train, self.y_train = np.array(self.x_train), np.array(
            self.y_train)  # Convert x_train and y_train to numpy arrays

        '''Reshape the data - lstm wants data to be 3-dimensional'''
        self.x_train = np.reshape(self.x_train, (
        self.x_train.shape[0], self.x_train.shape[1], self.x_train.shape[2]))  # shape of 232, 60, and 4 (4 columns)
        self.n_iters = 0

    def modelProcess(self, params):
        self.params = params  # Initialize params instance varable to store Bayesian Otimization derived search space parameters
        b_s, neuronPct, lr, dropout = self.params[0], self.params[1], self.params[2], self.params[
            3]  # Obtain desired search space params

        test_data = self.scaled_data[self.training_data_len - self.n_past:,:]  # Create the tesing data set & Create a new array scaled values from index
        self.x_test, self.y_test = [], self.dataset[self.training_data_len:,:]  # Create the data sets x_test and y_test as new instance variables
        for i in range(self.n_past, len(test_data)):
            self.x_test.append(test_data[i - self.n_past:i, :])

        self.x_test = np.array(self.x_test)  # Convert the data to numpy array
        self.x_test = np.reshape(self.x_test, (self.x_test.shape[0], self.x_test.shape[1], self.x_test.shape[2]))  # Reshape the data

        self.model = self.buildModel(self.x_train, neuronPct, lr, dropout)  # Call buildModel() definition to construct model using search space params
        history = self.model.fit(self.x_train, self.y_train, batch_size=int(b_s), epochs=10,
                                 validation_data=(self.x_test, self.y_test))  # Fit model

        '''Make predictions and compute Root Mean Square Error'''
        predictions = self.model.predict(self.x_test)
        prediction_copies = np.repeat(predictions, self.dataset.shape[1], axis=-1)
        predictions = self.scaler.inverse_transform(prediction_copies)[:, 0]
        rmse = np.sqrt(np.mean(predictions - self.y_test[:, 0]) ** 2)
        pctError = np.abs(np.mean((self.y_test[:, 0] - predictions) / self.y_test[:, 0])) * 100

        '''Make Forecast & Evaluate predictions'''
        self.forecast, self.train, self.valid = self.eval_predict(predictions, rmse, pctError, lr, b_s, dropout,
                                                                  self.training_data_len, self.dataFromDF, self.model)
        # eval_predict(predictions, rmse, lr, b_s, dropout, training_data_len, data, model)

        self.neuronCountList.append(neuronPct)
        self.batch_sizeList.append(b_s)
        self.learning_rateList.append(lr)
        self.dropoutList.append(dropout)
        self.rmseList.append(rmse)
        self.pctErrorList.append(pctError)
        return rmse

    def buildModel(self, x_train, nC, lr, dp):
        neuronCount = int(nC)
        # neuronShrink = nS
        model = Sequential()


        model.add(LSTM(neuronCount, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
        model.add(LSTM(neuronCount, return_sequences=False))

        model.add(Dense(0.5 * neuronCount))

        '''denseCount = neuronCount*neuronShrink
        if layers > 0:
            for i in range(layers):
              model.add(Dense(int(denseCount)))
              denseCount = denseCount*neuronShrink'''

        if dp > 0.0:
            model.add(Dropout(dp))
        model.add(Dense(1))

        opt = keras.optimizers.Adam(learning_rate=lr)  # Compile the model
        model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
        return model

    def eval_predict(self, predictions, rmse, pctError, lr, b_s, dp, training_data_len, data, model):
        print()
        print(self.state)
        print("RMSE Validation Score = " + str(rmse) + " | Percent Error: " + str(pctError))
        # print("RMSE Train Score = "+str(rmseForX_train))
        print("Learning Rate: " + str(lr))
        print("Batch Size: " + str(b_s))
        print("Dropout: " + str(dp))
        # self.rmse.append(rmse)

        '''Plot data'''
        train = data[:training_data_len]
        valid = data[training_data_len:]
        valid["Validation_Predictions"] = predictions.tolist()
        # valid["X_train_performance"] = x_trainPredictions.tolist()

        '''Visualize'''
        plt.figure(figsize=(16, 8))
        plt.title('Model Performance')
        plt.xlabel('Days since March 1, 2020')
        plt.ylabel('Cases')
        plt.plot([x for x in range(len(train['cases']))], train['cases'])
        plt.errorbar([x + len(train['cases']) for x in range(len(valid['cases']))], valid['Validation_Predictions'],
                     yerr=self.std)
        plt.plot([x + len(train['cases']) for x in range(len(valid['cases']))], valid['cases'])
        plt.plot([x + len(train['cases']) for x in range(len(valid['cases']))], valid['Validation_Predictions'])
        plt.legend(['Train Data (Actual)', 'Validation Data (Actual)', 'Validation Predictions (Predicted)'],
                   loc='lower right')
        plt.show()
        print(" ")

        '''Get the the first dataframe again'''
        df = self.df
        # Filter out features and store in datasets
        data = df.filter(['cases', "rates"])
        data2 = df.filter(['cases', "rates"])  # Copy for appending predictions'''

        n_future = 6
        testsForFuture = self.pplFullyVaccinated[-6:]
        casesRunningList = self.dailyCases
        future_forecast = []

        '''Predict next n_future days'''
        for i in range(n_future):
            last_60_days = data2[-self.n_past:].values
            last_60_days_scaled = self.scaler.transform(last_60_days)  # Scale the data to be value between 0 and 1
            X_test = []  # Create empty list and append past 60 days
            X_test.append(last_60_days_scaled)

            forPrediction = np.array(X_test)  # Convert the X_test data set to a numpy array
            forPrediction = np.reshape(forPrediction,
                                       (forPrediction.shape[0], forPrediction.shape[1], 2))  # Reshape data
            pred_newCase = model.predict(forPrediction)  # Get the predicted scaled cases
            pred_newCaseCopies = np.repeat(pred_newCase, 2, axis=-1)  # Repeat input data to fit scaler input_size
            pred_newCase = self.scaler.inverse_transform(pred_newCaseCopies)[:, 0]  # undo scaling
            future_forecast.append(pred_newCase[0])  # Append future_forecast for next one day
            casesRunningList.append(pred_newCase[0])
            derivatives = self.get_derivatives(casesRunningList)
            newData = pd.DataFrame({"cases": pred_newCase[0], "rates": derivatives[len(derivatives) - 1]}, index=[0])
            data1 = data2.append(newData)
        return future_forecast, train, valid

    '''Get Forecasts from Pipeline Object class'''
    def getForecasts(self):
        return self.forecast, self.train, self.valid

    def getStatistics(self):
        return self.std, self.variance

    '''Get Optimization Search Space params used throughout the training'''
    def getOptimizationSearchParams(self):
        return self.neuronCountList, self.batch_sizeList, self.learning_rateList, self.dropoutList, self.rmseList, self.pctErrorList

    '''Call save model method'''
    def saveModel(self, trialType):
        file_name = "shallowModel_" + trialType + "_" + self.state + ".h5"
        if os.path.isfile('C:/Users/surya/Documents/Data Science Projects/Covid-19 Analysis Project/Data/Model Data/' + file_name) is False:
            self.model.save('C:/Users/surya/Documents/Data Science Projects/Covid-19 Analysis Project/Data/Model Data/' + file_name)
        elif os.path.isfile('C:/Users/surya/Documents/Data Science Projects/Covid-19 Analysis Project/Data/Model Data/' + file_name) is True:
            os.remove('C:/Users/surya/Documents/Data Science Projects/Covid-19 Analysis Project/Data/Model Data/' + file_name)
            self.model.save('C:/Users/surya/Documents/Data Science Projects/Covid-19 Analysis Project/Data/Model Data/' + file_name)

    '''def __del__(self):
        print("Model Pipeline Instance for "+self.state+" Destructed")'''


# Long Model For OPtimization (Aka Intelli-Model) - Uses GPU configuration when specified (for faster training time) and trains using greater datasets
class long_opModel:
    def __init__(self, state, cases, deaths, rates, stateDates, pplFullVaccinated, govt_response, stateTesting):
        self.state = state

        '''Pull Github Data'''
        self.dailyCases = cases
        self.dailyDeaths = deaths
        self.ratesOfInfection = rates
        self.pplFullyVaccinated = pplFullVaccinated
        self.stateDates = stateDates
        self.govt_response = govt_response
        self.stateTesting = stateTesting

        '''Get Statistical Metrics'''
        self.mean = statistics.mean(self.dailyCases)
        self.variance = statistics.variance(self.dailyCases, xbar=self.mean)
        self.std = statistics.stdev(self.dailyCases, xbar=self.mean)

        '''Create slice dataframe for all data to be used'''
        self.data = {"date": self.stateDates,
                     "deaths": self.dailyDeaths,
                     "cases": self.dailyCases,
                     "rates": self.ratesOfInfection,
                     "pplFullyVaccinated": self.pplFullyVaccinated,
                     "govt_response": self.govt_response,
                     "stateTesting": self.stateTesting}

        self.batch_sizeList, self.neuronCountList, self.learning_rateList, self.dropoutList, self.rmseList, self.pctErrorList = [], [], [], [], [], []  # Initialize all search spaces
        self.df = pd.DataFrame(self.data, columns=["date", "cases", "rates", "pplFullyVaccinated", "govt_response", "stateTesting"])
        self.df.set_index('date', inplace=True)

        '''Filter out features and store in datasets'''
        self.dataFromDF = self.df.filter(["cases", "rates", "pplFullyVaccinated", "govt_response", "stateTesting"])
        self.datasetForTheLastTime = self.dataFromDF.values
        self.dataset = self.dataFromDF.values

        '''Specify Training Data length and number of observations to look at '''
        self.training_data_len = math.ceil(len(self.dataset) * .8)
        self.n_past = 30

        self.x_train, self.y_train, self.scaler, self.scaled_data = generate_scale(self.n_past,
                                                                                        self.training_data_len,
                                                                                        self.dataset)  # Scale and Normalize data
        self.x_train, self.y_train = np.array(self.x_train), np.array(
            self.y_train)  # Convert x_train and y_train to numpy arrays

        '''Reshape the data - lstm wants data to be 3-dimensional'''
        self.x_train = np.reshape(self.x_train, (
            self.x_train.shape[0], self.x_train.shape[1], self.x_train.shape[2]))  # shape of 232, 60, and 4 (4 columns)
        self.n_iters = 0

    def modelProcess(self, params):
        self.params = params  # Initialize params instance varable to store Bayesian Otimization derived search space parameters
        b_s, neuronPct, lr, dropout = self.params[0], self.params[1], self.params[2], self.params[3]  # Obtain desired search space params

        test_data = self.scaled_data[self.training_data_len - self.n_past:,:]  # Create the tesing data set & Create a new array scaled values from index
        self.x_test, self.y_test = [], self.dataset[self.training_data_len:,:]  # Create the data sets x_test and y_test as new instance variables
        for i in range(self.n_past, len(test_data)):
            self.x_test.append(test_data[i - self.n_past:i, :])

        self.x_test = np.array(self.x_test)  # Convert the data to numpy array
        self.x_test = np.reshape(self.x_test,
                                 (self.x_test.shape[0], self.x_test.shape[1], self.x_test.shape[2]))  # Reshape the data

        self.model = self.buildModel(self.x_train, neuronPct, lr,
                                     dropout)  # Call buildModel() definition to construct model using search space params
        history = self.model.fit(self.x_train, self.y_train, batch_size=int(b_s), epochs=10)  # Fit model

        '''Make predictions and compute Root Mean Square Error'''
        predictions = self.model.predict(self.x_test)
        prediction_copies = np.repeat(predictions, self.dataset.shape[1], axis=-1)
        predictions = self.scaler.inverse_transform(prediction_copies)[:, 0]
        rmse = np.sqrt(np.mean(predictions - self.y_test[:, 0]) ** 2)
        pctError = np.abs(np.mean((self.y_test[:, 0] - predictions) / self.y_test[:, 0])) * 100

        '''Make Forecast & Evaluate predictions'''
        self.forecast, self.train, self.valid, self.oneDayForecast = self.eval_predict(predictions, rmse, pctError, lr, b_s, dropout,
                                                                  self.training_data_len, self.dataFromDF, self.model)
        # eval_predict(predictions, rmse, lr, b_s, dropout, training_data_len, data, model)

        self.neuronCountList.append(neuronPct)
        self.batch_sizeList.append(b_s)
        self.learning_rateList.append(lr)
        self.dropoutList.append(dropout)
        self.rmseList.append(rmse)
        self.pctErrorList.append(pctError)
        # print(rmse+"/n")
        return rmse

    def buildModel(self, x_train, nC, lr, dp):
        neuronCount = int(nC)
        # neuronShrink = nS
        model = Sequential()

        model.add(LSTM(neuronCount, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
        model.add(LSTM(neuronCount, return_sequences=False))

        model.add(Dense(0.5 * neuronCount))

        '''denseCount = neuronCount*neuronShrink
        if layers > 0:
            for i in range(layers):
              model.add(Dense(int(denseCount)))
              denseCount = denseCount*neuronShrink'''

        if dp > 0.0:
            model.add(Dropout(dp))
        model.add(Dense(1))

        opt = keras.optimizers.Adam(learning_rate=lr)  # Compile the model
        model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
        return model

    def eval_predict(self, predictions, rmse, pctError, lr, b_s, dp, training_data_len, data, model):
        print()
        print("---- Long Model Optimization ----")
        print("RMSE Validation Score = " + str(rmse) + " | Percent Error: " + str(pctError))
        print("Learning Rate: " + str(lr) + " | Batch Size: " + str(b_s) + " | Dropout: " + str(dp))

        '''Plot data'''
        train = data[:training_data_len]
        valid = data[training_data_len:]
        valid["Validation_Predictions"] = predictions.tolist()

        '''Visualize'''
        plt.figure(figsize=(16, 8))
        plt.title('Model Performance')
        plt.xlabel('Days since March 1, 2020')
        plt.ylabel('Cases')
        plt.plot([x for x in range(len(train['cases']))], train['cases'])
        plt.errorbar([x + len(train['cases']) for x in range(len(valid['cases']))], valid['Validation_Predictions'],
                     yerr=self.std)
        plt.plot([x + len(train['cases']) for x in range(len(valid['cases']))], valid['cases'])
        plt.plot([x + len(train['cases']) for x in range(len(valid['cases']))], valid['Validation_Predictions'])
        plt.legend(['Train Data (Actual)', 'Validation Data (Actual)', 'Validation Predictions (Predicted)'],
                   loc='lower right')
        plt.show()
        print(" /n")

        '''Get the the first dataframe again'''
        df = self.df
        # Filter out features and store in datasets
        data = df.filter(["cases", "rates", "pplFullyVaccinated", "govt_response", "stateDailyTesting"])
        data2 = self.dataFromDF

        if(len(self.dailyCases) > len(self.ratesOfInfection)):
            casesRunningList = self.dailyCases[:-7]
            self.dailyCases = casesRunningList
        else:
            casesRunningList = self.dailyCases

        future_forecast = []
        futureCasesVar = 0

        last_60_days = data2[-self.n_past:].values
        print(last_60_days.shape)
        print(self.scaled_data.shape)
        #print(self.scaler)
        last_60_days_scaled = self.scaler.transform(last_60_days)  # Scale the data to be value between 0 and 1
        X_test = []  # Create empty list and append past 60 days
        X_test.append(last_60_days_scaled)

        forPrediction = np.array(X_test)  # Convert the X_test data set to a numpy array
        forPrediction = np.reshape(forPrediction, (forPrediction.shape[0], forPrediction.shape[1], 5))  # Reshape data '''------Change that 5 to an actual code '''
        pred_newCase = model.predict(forPrediction)  # Get the predicted scaled cases
        pred_newCaseCopies = np.repeat(pred_newCase, 5, axis=-1)  # Repeat input data to fit scaler input_size
        pred_newCase = self.scaler.inverse_transform(pred_newCaseCopies)[:, 0]  # undo scaling
        futureCasesVar = pred_newCase[0]
        future_forecast.append(pred_newCase[0])  # future_forecast for next one day
        casesRunningList.append(pred_newCase[0])
        derivatives = self.get_derivatives(casesRunningList)

        future_trend_predictor = shallow_model(self.state, casesRunningList, [], derivatives, self.stateDates, [], "optimization")
        rmseForPredictor = future_trend_predictor.modelProcess(b_s)
        forecast, trainData, ValidData = future_trend_predictor.getForecasts()
        future_forecast+=forecast
        return future_forecast, train, valid, futureCasesVar

    '''Get Forecasts from Pipeline Object class'''
    def getForecasts(self):
        return self.forecast, self.train, self.valid, self.oneDayForecast

    def getStatistics(self):
        return self.std, self.variance

    '''Get Optimization Search Space params used throughout the training'''
    def getOptimizationSearchParams(self):
        return self.neuronCountList, self.batch_sizeList, self.learning_rateList, self.dropoutList, self.rmseList, self.pctErrorList

    '''Call save model method'''
    def saveModel(self, trialType):
        file_name = "longModel_" + trialType + "_" + self.state + ".h5"
        if os.path.isfile('C:/Users/surya/Documents/Data Science Projects/Covid-19 Analysis Project/Data/Model Data/' + file_name) is False:
            self.model.save('C:/Users/surya/Documents/Data Science Projects/Covid-19 Analysis Project/Data/Model Data/' + file_name)
        elif os.path.isfile('C:/Users/surya/Documents/Data Science Projects/Covid-19 Analysis Project/Data/Model Data/' + file_name) is True:
            os.remove('C:/Users/surya/Documents/Data Science Projects/Covid-19 Analysis Project/Data/Model Data/' + file_name)
            self.model.save('C:/Users/surya/Documents/Data Science Projects/Covid-19 Analysis Project/Data/Model Data/' + file_name)

    '''def __del__(self):
        print("Model Pipeline Instance for "+self.state+" Destructed")'''


# Shallow-Model Case Predictor - For Model Reconstruction and Prediction
class shallow_model:
    def __init__(self, state, cases, deaths, rates, stateDates, pplFullVaccinated, runType="prediction"):
        self.state = state

        '''Pull Github Data'''
        self.dailyCases = cases
        self.dailyDeaths = deaths
        self.ratesOfInfection = rates
        self.pplFullyVaccinated = pplFullVaccinated
        self.stateDates = stateDates

        '''Get Statistical Metrics'''
        self.mean = statistics.mean(self.dailyCases)
        self.variance = statistics.variance(self.dailyCases, xbar=self.mean)
        self.std = statistics.stdev(self.dailyCases, xbar=self.mean)

        if(runType == "optimization"):
            stateDatesList = [x for x in self.stateDates]
            stateDatesList += ['2021-05-02']
            # print(len(self.dailyCases.get(state)))
            '''Create slice dataframe for all data to be used'''
            self.data = {"date": stateDatesList,
                     "deaths": self.dailyDeaths,
                     "cases": self.dailyCases,
                     "rates": self.ratesOfInfection,
                     "pplFullyVaccinated": self.pplFullyVaccinated}
        else:
            self.data = {"date": self.stateDates,
                         "deaths": self.dailyDeaths,
                         "cases": self.dailyCases,
                         "rates": self.ratesOfInfection,
                         "pplFullyVaccinated": self.pplFullyVaccinated}

        self.batch_sizeList, self.neuronCountList, self.learning_rateList, self.dropoutList, self.rmseList, self.pctErrorList = [], [], [], [], [], []  # Initialize all search spaces
        self.df = pd.DataFrame(self.data, columns=["date", "cases", "rates"])
        self.df.set_index('date', inplace=True)

        '''Filter out features and store in datasets'''
        self.dataFromDF = self.df.filter(["cases", "rates"])
        self.datasetForTheLastTime = self.dataFromDF.values
        self.dataset = self.dataFromDF.values

        '''Specify Training Data length and number of observations to look at '''
        self.training_data_len = math.ceil(len(self.dataset) * .8)
        self.n_past = 30

        print(len(self.dataset))
        self.x_train, self.y_train, self.scaler, self.scaled_data = generate_scale(self.n_past,
                                                                                        self.training_data_len,
                                                                                        self.dataset)  # Scale and Normalize data
        self.x_train, self.y_train = np.array(self.x_train), np.array(
            self.y_train)  # Convert x_train and y_train to numpy arrays

        '''Reshape the data - lstm wants data to be 3-dimensional'''
        self.x_train = np.reshape(self.x_train, (
        self.x_train.shape[0], self.x_train.shape[1], self.x_train.shape[2]))  # shape of 232, 60, and 4 (4 columns)
        self.n_iters = 0

    def modelProcess(self, b_s):
        test_data = self.scaled_data[self.training_data_len - self.n_past:,:]  # Create the tesing data set & Create a new array scaled values from index
        self.x_test, self.y_test = [], self.dataset[self.training_data_len:,
                                       :]  # Create the data sets x_test and y_test as new instance variables
        for i in range(self.n_past, len(test_data)):
            self.x_test.append(test_data[i - self.n_past:i, :])

        self.x_test = np.array(self.x_test)  # Convert the data to numpy array
        self.x_test = np.reshape(self.x_test,
                                 (self.x_test.shape[0], self.x_test.shape[1], self.x_test.shape[2]))  # Reshape the data

        self.model = self.buildModel(self.x_train)  # Call buildModel() definition to construct model using search space params
        history = self.model.fit(self.x_train, self.y_train, batch_size=int(b_s), epochs=10,
                                 validation_data=(self.x_test, self.y_test))  # Fit model

        '''Make predictions and compute Root Mean Square Error'''
        predictions = self.model.predict(self.x_test)
        prediction_copies = np.repeat(predictions, self.dataset.shape[1], axis=-1)
        predictions = self.scaler.inverse_transform(prediction_copies)[:, 0]
        rmse = np.sqrt(np.mean(predictions - self.y_test[:, 0]) ** 2)
        pctError = np.abs(np.mean((self.y_test[:, 0] - predictions) / self.y_test[:, 0])) * 100

        '''Make Forecast & Evaluate predictions'''
        self.forecast, self.train, self.valid = self.eval_predict(predictions, rmse, pctError,
                                                                  self.training_data_len, self.dataFromDF, self.model)

        self.rmseList.append(rmse)
        self.pctErrorList.append(pctError)
        return rmse

    def buildModel(self, x_train):
        file_name = "shallowModel_" + "cases" + "_" + self.state + ".h5"  #change the cases to a parameter called trialType later
        if os.path.isfile('C:/Users/surya/Documents/Data Science Projects/Covid-19 Analysis Project/Data/Model Data/' + file_name) is True:
            model = tf.keras.model.load_model('C:/Users/surya/Documents/Data Science Projects/Covid-19 Analysis Project/Data/Model Data/' + file_name)
        else:
            raise Exception("No Optimized File Implementation Created or Provided for use")
        return model

    def eval_predict(self, predictions, rmse, pctError, training_data_len, data, model):
        print()
        print("---- Shallow Model Prediction ----")
        print("RMSE Validation Score = " + str(rmse) + " | Percent Error: " + str(pctError))

        '''Plot data'''
        train = data[:training_data_len]
        valid = data[training_data_len:]
        valid["Validation_Predictions"] = predictions.tolist()

        '''Visualize'''
        plt.figure(figsize=(16, 8))
        plt.title('Model Performance')
        plt.xlabel('Days since March 1, 2020')
        plt.ylabel('Cases')
        plt.plot([x for x in range(len(train['cases']))], train['cases'])
        plt.errorbar([x + len(train['cases']) for x in range(len(valid['cases']))], valid['Validation_Predictions'],
                     yerr=self.std)
        plt.plot([x + len(train['cases']) for x in range(len(valid['cases']))], valid['cases'])
        plt.plot([x + len(train['cases']) for x in range(len(valid['cases']))], valid['Validation_Predictions'])
        plt.legend(['Train Data (Actual)', 'Validation Data (Actual)', 'Validation Predictions (Predicted)'],
                   loc='lower right')
        plt.show()
        print(" /n")

        '''Get the the first dataframe again'''
        df = self.df
        # Filter out features and store in datasets
        data = df.filter(['cases', "rates"])
        data2 = df.filter(['cases', "rates"])  # Copy for appending predictions'''

        n_future = 6
        testsForFuture = self.pplFullyVaccinated[-6:]
        casesRunningList = self.dailyCases
        future_forecast = []

        '''Predict next n_future days'''
        for i in range(n_future):
            last_60_days = data2[-self.n_past:].values
            last_60_days_scaled = self.scaler.transform(last_60_days)  # Scale the data to be value between 0 and 1
            X_test = []  # Create empty list and append past 60 days
            X_test.append(last_60_days_scaled)

            forPrediction = np.array(X_test)  # Convert the X_test data set to a numpy array
            forPrediction = np.reshape(forPrediction,
                                       (forPrediction.shape[0], forPrediction.shape[1], 2))  # Reshape data
            pred_newCase = model.predict(forPrediction)  # Get the predicted scaled cases
            pred_newCaseCopies = np.repeat(pred_newCase, 2, axis=-1)  # Repeat input data to fit scaler input_size
            pred_newCase = self.scaler.inverse_transform(pred_newCaseCopies)[:, 0]  # undo scaling
            future_forecast.append(pred_newCase[0])  # Append future_forecast for next one day
            casesRunningList.append(pred_newCase[0])
            derivatives = self.get_derivatives(casesRunningList)
            newData = pd.DataFrame({"cases": pred_newCase[0], "rates": derivatives[len(derivatives) - 1]}, index=[0])
            data1 = data2.append(newData)
        return future_forecast, train, valid

    '''Get Forecasts from Pipeline Object class'''

    def getForecasts(self):
        return self.forecast, self.train, self.valid

    def getStatistics(self):
        return self.std, self.variance

    '''Get Optimization Search Space params used throughout the training'''

    def getErrors(self):
        return self.rmseList, self.pctErrorList

    '''def __del__(self):
        print("Model Pipeline Instance for "+self.state+" Destructed")'''


# Long Model for predictions
class long_model:
    def __init__(self, state, cases, deaths, rates, pplFullVaccinated, stateDates, govt_response,
                 stateTesting, predictionType, desiredRange, targetCase=-1):
        self.state = state
        self.desiredRange = desiredRange
        self.predictionType = predictionType
        if(targetCase >= 0):
            self.targetCase = targetCase
        else:
            self.targetCase = 0

        '''Pull Github Data'''
        self.dailyCases = cases[:desiredRange]
        self.dailyDeaths = deaths[:desiredRange]
        self.ratesOfInfection = rates[:desiredRange]
        self.pplFullyVaccinated = pplFullVaccinated[:desiredRange]
        self.stateDates = stateDates[:desiredRange]
        self.govt_response = govt_response[:desiredRange]
        self.stateTesting = stateTesting[:desiredRange]

        self.n_past = 30

        '''Get Statistical Metrics'''
        self.mean = statistics.mean(self.dailyCases)
        self.variance = statistics.variance(self.dailyCases, xbar=self.mean)
        self.std = statistics.stdev(self.dailyCases, xbar=self.mean)

    def modelProcess(self, params):
        self.params = params  # Initialize params instance varable to store Bayesian Otimization derived search space parameters
        pplFullyVaccinatedIn, govt_stingencyIn, stateTestingIn = self.params[0], self.params[1], self.params[2] # Obtain desired search space params

        '''Create slice dataframe for all data to be used'''
        self.pplFullyVaccinated[-1] = pplFullyVaccinatedIn
        self.govt_response[-1] = govt_stingencyIn
        self.stateTesting[-1] = stateTestingIn
        self.data = {"date": self.stateDates[-self.n_past:],
                     "deaths": self.dailyDeaths[-self.n_past:],
                     "cases": self.dailyCases[-self.n_past:],
                     "rates": self.ratesOfInfection[-self.n_past:],
                     "pplFullyVaccinated": self.pplFullyVaccinated[-self.n_past:],
                     "govt_response": self.govt_response[-self.n_past:],
                     "stateTesting": self.stateTesting[-self.n_past:]}

        self.df = pd.DataFrame(self.data, columns=["date", "cases", "rates", "pplFullyVaccinated", "govt_response",
                                                   "stateTesting"])
        self.df.set_index('date', inplace=True)

        '''Filter out features and store in datasets'''
        self.dataFromDF = self.df.filter(["cases", "rates", "pplFullyVaccinated", "govt_response", "stateTesting"])
        self.datasetForTheLastTime = self.dataFromDF.values
        self.dataset = self.dataFromDF.values

        self.training_data_len = math.ceil(len(self.dataset) * 1.0)

        '''Specify Training Data length and number of observations to look at '''
        self.training_data_len = math.ceil(len(self.dataset) * .8)

        self.x, self.scaler, self.scaled_data = generate_scale(self.n_past, self.training_data_len, self.dataset)  # Scale and Normalize data
        self.x = np.array(self.x)  # Convert x_train and y_train to numpy arrays'''

        '''Reshape the data - lstm wants data to be 3-dimensional'''
        '''self.x_train = np.reshape(self.x_train, (
        self.x_train.shape[0], self.x_train.shape[1], self.x_train.shape[2]))  # shape of 232, 60, and 4 (4 columns)'''

        self.x = np.reshape(self.x, (
            self.x.shape[0], self.x.shape[1], self.x.shape[2]))  # shape of 232, 60, and 4 (4 columns)

        self.n_iters = 0


        self.model = self.buildModel(self.x)  # Call buildModel() definition to construct model using search space params

        '''Make predictions and compute Root Mean Square Error'''
        predictions = self.model.predict(self.x)
        prediction_copies = np.repeat(predictions, self.dataset.shape[1], axis=-1)
        predictions = self.scaler.inverse_transform(prediction_copies)[:, 0]


        '''Make Forecast & Evaluate predictions'''
        self.forecast, self.train, self.valid, self.oneDayForecast = self.eval_predict(predictions,
                                                self.training_data_len, self.dataFromDF, self.model)

        if(self.predictionType == "minimize"):
            return self.oneDayForecast
        if(self.predictionType == "toValue"):
            return math.abs(self.oneDayForecast-self.targetCase)
        else:
            return predictions

    def buildModel(self, x_train):
        file_name = "longModel_" + "cases" + "_" + self.state + ".h5"
        if os.path.isfile(
                'C:/Users/surya/Documents/Data Science Projects/Covid-19 Analysis Project/Data/Model Data/' + file_name) is True:
            model = tf.keras.models.load_model('C:/Users/surya/Documents/Data Science Projects/Covid-19 Analysis Project/Data/Model Data/' + file_name)
        else:
            raise Exception("No Optimized File Implementation Created or Provided for use")
        return model

    def eval_predict(self, predictions, rmse, pctError, b_s, training_data_len, data, model):
        print()
        print("RMSE Validation Score = " + str(rmse) + " | Percent Error: " + str(pctError))

        '''Plot data'''
        train = data[:training_data_len]
        valid = data[training_data_len:]
        valid["Validation_Predictions"] = predictions.tolist()

        '''Visualize'''
        plt.figure(figsize=(16, 8))
        plt.title('Model Performance')
        plt.xlabel('Days since March 1, 2020')
        plt.ylabel('Cases')
        plt.plot([x for x in range(len(train['cases']))], train['cases'])
        plt.errorbar([x + len(train['cases']) for x in range(len(valid['cases']))], valid['Validation_Predictions'],
                     yerr=self.std)
        plt.plot([x + len(train['cases']) for x in range(len(valid['cases']))], valid['cases'])
        plt.plot([x + len(train['cases']) for x in range(len(valid['cases']))], valid['Validation_Predictions'])
        plt.legend(['Train Data (Actual)', 'Validation Data (Actual)', 'Validation Predictions (Predicted)'],
                   loc='lower right')
        plt.show()
        print(" /n")

        '''Get the the first dataframe again'''
        df = self.df
        # Filter out features and store in datasets
        data = df.filter(["cases", "rates", "pplFullyVaccinated", "govt_response", "stateDailyTesting"])
        data2 = df.filter(["cases", "rates", "pplFullyVaccinated", "govt_response",
                           "stateDailyTesting"])  # Copy for appending predictions'''

        testsForFuture = self.pplFullyVaccinated.get(self.state)[-6:]
        testingForFuture = self.stateDailyTesting.get(self.state)[-6:]
        casesRunningList = self.dailyCases.get(self.state)
        future_forecast = []
        futureCasesDay1 = 0;

        last_60_days = data2[-self.n_past:].values
        last_60_days_scaled = self.scaler.transform(last_60_days)  # Scale the data to be value between 0 and 1
        X_test = []  # Create empty list and append past 60 days
        X_test.append(last_60_days_scaled)

        forPrediction = np.array(X_test)  # Convert the X_test data set to a numpy array
        forPrediction = np.reshape(forPrediction, (forPrediction.shape[0], forPrediction.shape[1], 5))  # Reshape data '''------Change that 5 to an actual code '''
        pred_newCase = model.predict(forPrediction)  # Get the predicted scaled cases
        pred_newCaseCopies = np.repeat(pred_newCase, 2, axis=-1)  # Repeat input data to fit scaler input_size
        pred_newCase = self.scaler.inverse_transform(pred_newCaseCopies)[:, 0]  # undo scaling
        futureCasesDay1 = pred_newCase[0]
        future_forecast.append(futureCasesDay1)  # future_forecast for next one day
        casesRunningList.append(futureCasesDay1)
        derivatives = self.get_derivatives(casesRunningList)
        predictor = shallow_model(self.state, casesRunningList, [], derivatives, [], [])
        rmseForPredictor = predictor.modelProcess(b_s)
        forecast, trainData, ValidData = predictor.getForecasts()
        future_forecast+=forecast

        return future_forecast, train, valid, futureCasesDay1

    '''Get Forecasts from Pipeline Object class'''
    def getForecasts(self):
        return self.forecast, self.train, self.valid, self.oneDayForecast

    def getStatistics(self):
        return self.std, self.variance

    '''Get Optimization Search Space params used throughout the training'''
    def getErrors(self):
        return self.rmseList, self.pctErrorList


    '''def __del__(self):
        print("Model Pipeline Instance for "+self.state+" Destructed")'''