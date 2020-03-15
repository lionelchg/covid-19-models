########################################################################################################################
#                                                                                                                      #
#                                       Covid-19-metrics: data_processing module                                       #
#                                                                                                                      #
#                                               Lionel Cheng 15.03.2020                                                #
#                                                                                                                      #
########################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.linear_model import LinearRegression

dict_keys = ['Province/State', 'Country/Region']

def read_data(filename):
    """Reads data from the CSSE database (John Hopkins)"""
    cases = []
    with open(filename, 'r') as csv_file:
        rows = csv.reader(csv_file, delimiter=',')
        row_count = 0
        for row in rows:
            if row_count==0:
                header = row
                start_date = row[4]
                ndays = len(row[4:])
            else:
                if row[0].split():
                    if row[0].split()[-1][-1].isupper():
                        break
                country = []
                country.append({dict_keys[i]:row[i] for i in range(2)})
                country.append(np.array(row[4:], dtype=np.float32))
                cases.append(country)
            row_count += 1

    regions_count = row_count - 1

    total_cases = np.zeros(ndays)
    total_out_china = np.zeros(ndays)
    total_china = np.zeros(ndays)

    for region in cases:
        total_cases += region[1]
        if region[0][dict_keys[1]] == 'China':
            total_china += region[1]
        else:
            total_out_china += region[1]

    return cases, total_cases, total_china, total_out_china

def retrieve_region(region_str, cases, ndays, exclude=False):
    """Extracts data from a region"""
    cases_tmp = np.zeros(ndays)

    for region in cases:
        if exclude:
            if region[0][dict_keys[1]] != region_str:
                cases_tmp += region[1]
        else:
            if region[0][dict_keys[1]] == region_str:
                cases_tmp += region[1]

    return cases_tmp

def log_reglin(days, cases):
    """Exponential model with an estimate of the total cases, 
    start of the model with at least 30 people infected"""
    for index_min in range(len(days)):
        if cases[index_min] >= 30:
            break
    log_cases = np.log(cases[index_min:]).reshape(-1, 1)
    days = days[index_min:].reshape(-1, 1)
    reg = LinearRegression().fit(days, log_cases)

    R = reg.score(days, log_cases)
    a = reg.coef_
    b = reg.intercept_
    alpha = np.exp(a) - 1

    cases_model = np.exp(a * days + b)

    return cases_model, R, a, b, alpha, days

def logistic_model(days, alpha, ntot, cases):
    """Logistic model with an estimate of the total cases, 
    start of the model with at least 30 people infected"""
    for index_min in range(len(days)):
        if cases[index_min] >= 30:
            break
    model_days = days[index_min:]
    ndays = len(model_days)
    cases_model = np.zeros(ndays)
    cases_model[0] = cases[index_min]

    for i in range(1, ndays):
        cases_model[i] = (1 + alpha * (1 - cases_model[i-1] / ntot)) * cases_model[i-1]
    return cases_model, model_days

def evolution(days, cases):
    """Evolution metrics, first derivative (increase),
    the growth factor is linked to the second derivative and is 
    above one when the function is convex and concave otherwise"""
    ndays = len(days)
    increase = np.zeros(ndays)
    increase[1:-1] = (cases[2:] - cases[:-2]) / 2
    increase[0] = (- 3 * cases[0] + 4 * cases[1] - cases[2]) / 2
    increase[-1] = (3 * cases[-1] - 4 * cases[-2] + cases[-3]) / 2

    growth_factor = np.zeros(ndays)
    growth_factor[1:] = increase[1:] / increase[:-1]
    growth_factor[0:2] = 0

    return increase, growth_factor