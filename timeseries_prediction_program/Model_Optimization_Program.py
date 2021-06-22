import os.path
import time
import seaborn as sns
import sys

#Basic Data Science Frameworks
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import kendalltau
from scipy.stats import spearmanr
from scipy.stats import pointbiserialr
from pandas import DataFrame
from pandas import concat
import statistics

#Scipy libraries (Piece-wise Interp)
from scipy.interpolate import splev, splrep
from scipy.optimize import curve_fit

#Utilities
from datetime import datetime
import math
import warnings
warnings.filterwarnings("ignore")

#!pip install scikit-optimize
from skopt import gp_minimize
from skopt.plots import plot_convergence
from skopt.plots import plot_objective

import simplejson as json
import requests

#import Shallow_Covid19_Case_Predictor as s_model
from PFE_predictors import shallow_opModel as s_model
from PFE_predictors import long_opModel as l_model

#longRun = False

def dataExtraction():
    dailyCases = json.loads(
        requests.get('https://raw.githubusercontent.com/surya2/Covid-19_Analysis_ML/main/dailyCases.json').text)
    dailyDeaths = json.loads(
        requests.get('https://raw.githubusercontent.com/surya2/Covid-19_Analysis_ML/main/dailyDeaths.json').text)
    ratesOfInfection = json.loads(
        requests.get('https://raw.githubusercontent.com/surya2/Covid-19_Analysis_ML/main/ratesOfInfection.json').text)
    pplFullyVaccinated = json.loads(requests.get(
        'https://raw.githubusercontent.com/surya2/Covid-19_Analysis_ML/main/peopleFullyVaccinatedPer100.json').text)
    stateDates = json.loads(
        requests.get('https://raw.githubusercontent.com/surya2/Covid-19_Analysis_ML/main/stateDates.json').text)
    stateTimestamps = json.loads(
        requests.get('https://raw.githubusercontent.com/surya2/Covid-19_Analysis_ML/main/stateTimestamps.json').text)

    stateWithLongestPandemic = ""
    longestPandemic = 300
    for i in dailyCases.keys():
        if (len(stateDates.get(i)) > longestPandemic):
            longestPandemic = len(stateDates.get(i))
            stateWithLongestPandemic = i

    '''Normalize all datasets within a constant interval'''
    for s in stateDates.keys():
        differenceInLen = longestPandemic - len(stateDates.get(s))
        if (differenceInLen > 0):
            stateDates[s] = stateDates.get(stateWithLongestPandemic)[:differenceInLen] + stateDates.get(s)
            stateTimestamps[s] = stateTimestamps.get(stateWithLongestPandemic)[:differenceInLen] + stateTimestamps.get(
                s)
            dailyCases[s] = [0] * differenceInLen + dailyCases.get(s)
            dailyDeaths[s] = [0] * differenceInLen + dailyDeaths.get(s)
            ratesOfInfection[s] = [0] * differenceInLen + ratesOfInfection.get(s)
            pplFullyVaccinated[s] = [0] * differenceInLen + pplFullyVaccinated.get(s)

    '''Prepare Vaccination Dataset'''
    vaccinationPhaseData = []
    for s in stateDates.keys():
        priorVaccinationData = pplFullyVaccinated.get(s)[:316]
        for j in range(len(pplFullyVaccinated.get(s)[316])):
            vaccinationPhaseData.append(pplFullyVaccinated.get(s)[316][j])
        pplFullyVaccinated[s] = priorVaccinationData + vaccinationPhaseData
        vaccinationPhaseData = []

    return dailyCases, dailyDeaths, ratesOfInfection, pplFullyVaccinated, stateDates, stateTimestamps

def defineData(dataRequest="shallow"):
    if (dataRequest == "shallow"):
        return dataExtraction()

    if (dataRequest == "long"):
        govt_response = json.loads(requests.get('https://raw.githubusercontent.com/surya2/Covid-19_Analysis_ML/main/govt_response_index.json').text)
        stateTesting = json.loads(requests.get('https://raw.githubusercontent.com/surya2/Covid-19_Analysis_ML/main/state_testing.json').text)

        dailyCases, dailyDeaths, ratesOfInfection, pplFullyVaccinated, stateDates, stateTimestamps = dataExtraction()

        stateDailyTesting = {}
        for s in stateDates.keys():
            dailyTesting = []
            dayTest = 0
            for i in range(len(stateTesting.get(s))):
                if (i == 0):
                    dayTest = float(stateTesting.get(s)[i])
                    if (stateTesting.get(s)[i] == "nan" or math.isnan(dayTest)):
                        dayTest = 0
                elif (stateTesting.get(s)[i] == "nan"):
                    dayTest = 0
                else:
                    dayTests = float(stateTesting.get(s)[i]) - float(stateTesting.get(s)[i - 1])
                    if (dayTest < 0 or math.isnan(dayTest)):
                        dayTest = 0

                dailyTesting.append(dayTest)
            stateDailyTesting[s] = dailyTesting
        return dailyCases, dailyDeaths, ratesOfInfection, pplFullyVaccinated, stateDates, stateTimestamps, govt_response, stateDailyTesting

    # state = "Virginia" #figure a way to change this paramtre to construct multiple state-specific models
    optimizationModel = None

def optimizeHelperFunc(params):
    rmse = optimizationModel.modelProcess(params)
    '''if(longRun == True):
        dailyCases, dailyDeaths, ratesOfInfection, pplFullyVaccinated, stateDates, stateTimestamps, govt_response, stateTesting = defineData("long")
        optimizationModel.set_params(dailyCases, dailyDeaths, ratesOfInfection, pplFullyVaccinated, stateDates, stateTimestamps, govt_response, stateTesting)
    '''
    return rmse

t0 = time.clock()
rmseListStates, pctErrorStates, learningRateStates, neuronCountStates, batchSizeStates, dropoutStates = {}, {}, {}, {}, {}, {}
forecastsStates, trainDataStates, validDataStates = {}, {}, {}
stdStates, varianceStates = {}, {}
searchSpaces = {}

for state in ["Virginia", "New York", "New Jersey", "Delaware", "Maryland", "Pennsylvania", "West Virginia"]:
    dailyCases, dailyDeaths, ratesOfInfection, pplFullyVaccinated, stateDates, stateTimestamps = defineData()
    optimizationModel = s_model(state, dailyCases.get(state), dailyDeaths.get(state), ratesOfInfection.get(state),
                                stateDates.get(state), pplFullyVaccinated.get(state))

    search_space = [(2, 32), (2, 100), (0.00001, 0.1), (0.0, 0.499)]

    r = gp_minimize(optimizeHelperFunc, search_space, n_calls=50, random_state=1)
    # print(r.x+"/n"+r.fun)
    neuronCountStates[state], batchSizeStates[state], learningRateStates[state], dropoutStates[state], \
    rmseListStates[state], pctErrorStates[state] = optimizationModel.getOptimizationSearchParams()
    del optimizationModel
    time.sleep(3)
    print(" ")
    print("Optimized Model Training and Forecast Collection")
    dailyCases, dailyDeaths, ratesOfInfection, pplFullyVaccinated, stateDates, stateTimestamps = defineData()
    optimizedModel = s_model(state, dailyCases.get(state), dailyDeaths.get(state), ratesOfInfection.get(state),
                             stateDates.get(state), pplFullyVaccinated.get(state))
    searchSpaces[state] = r.x
    rmse = optimizedModel.modelProcess(searchSpaces.get(state))
    forecastsStates[state], trainDataStates[state], validDataStates[state] = optimizedModel.getForecasts()
    stdStates[state], varianceStates[state] = optimizedModel.getStatistics()
    optimizedModel.saveModel("cases")
    del optimizedModel
print("Wall Time for shallow model optimization: "+time.clock()-t0+" seconds process time")

t0 = time.clock()
rmseListStatesLong, pctErrorStatesLong, learningRateStatesLong, neuronCountStatesLong, batchSizeStatesLong, dropoutStatesLong = {}, {}, {}, {}, {}, {}
forecastsStatesLong, trainDataStatesLong, validDataStatesLong, oneDayForecastStates = {}, {}, {}, {}
stdStatesLong, varianceStatesLong = {}, {}
searchSpacesLong = {}

for state in ["Virginia", "New York", "New Jersey", "Delaware", "Maryland", "Pennsylvania", "West Virginia"]:
    dailyCases, dailyDeaths, ratesOfInfection, pplFullyVaccinated, stateDates, stateTimestamps, govt_response, stateTesting = defineData(
        "long")
    optimizationModel = l_model(state, dailyCases.get(state), dailyDeaths.get(state), ratesOfInfection.get(state),
                                stateDates.get(state), pplFullyVaccinated.get(state), govt_response.get(state),
                                stateTesting.get(state))

    search_space = [(2, 32), (2, 100), (0.00001, 0.1), (0.0, 0.499)]

    r = gp_minimize(optimizeHelperFunc, search_space, n_calls=50, random_state=1)
    # print(r.x+"/n"+r.fun)
    neuronCountStatesLong[state], batchSizeStatesLong[state], learningRateStatesLong[state], dropoutStatesLong[state], \
    rmseListStatesLong[state], pctErrorStatesLong[state] = optimizationModel.getOptimizationSearchParams()

    del optimizationModel
    time.sleep(3)
    print(" ")
    print("Optimized Model Training and Forecast Collection")
    dailyCases, dailyDeaths, ratesOfInfection, pplFullyVaccinated, stateDates, stateTimestamps, govt_response, stateTesting = defineData(
        "long")
    optimizedModel = l_model(state, dailyCases.get(state), dailyDeaths.get(state), ratesOfInfection.get(state),
                             stateDates.get(state), pplFullyVaccinated.get(state), govt_response.get(state),
                             stateTesting.get(state))
    searchSpacesLong[state] = r.x
    rmse = optimizedModel.modelProcess(searchSpacesLong.get(state))
    forecastsStatesLong[state], trainDataStatesLong[state], validDataStatesLong[state], oneDayForecastStates[
        state] = optimizedModel.getForecasts()
    stdStatesLong[state], varianceStatesLong[state] = optimizedModel.getStatistics()
    optimizedModel.saveModel("cases")
    del optimizedModel
print("Wall time for long model optimization: "+time.clock()-t0+" seconds process time")



