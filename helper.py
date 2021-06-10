import numpy as np
import scipy
import dateutil.parser
import datetime
from collections import defaultdict
import json


def delay(actual_date, days_to_shift):
    return scipy.ndimage.interpolation.shift(actual_date, days_to_shift)

def pre_processing():
    population_dict = defaultdict()
    f1 = open('country-by-population.json')
    data = json.load(f1)
    for l in data:
        country = l['country']
        population = l['population']
        population_dict[country] = population
    f2 = open('covid-19_data.json')
    real_data = json.load(f2)
    return population_dict, real_data

def get_country_xcdr(country, real_data):
    data_dict = defaultdict()
    all_date = []
    result = []
    for i, location in enumerate(real_data['confirmed']['locations']):
        if country != location['country']:
            continue
        for date in location['history']:
            try:
                cur_confirmed = int(location['history'][date])
            except (KeyError, IndexError):
                cur_confirmed = 0
            try:
                cur_deaths = int(real_data['deaths']['locations'][i]['history'][date])
            except (KeyError, IndexError):
                cur_deaths = 0
            try:
                cur_recovered = int(real_data['recovered']['locations'][i]['history'][date])
            except (KeyError, IndexError):
                cur_recovered = 0
            cur_date = dateutil.parser.parse(date)
            if cur_date not in data_dict:
                data_dict[cur_date] = [0, 0, 0]
            data_dict[cur_date][0] += cur_confirmed
            data_dict[cur_date][1] += cur_recovered
            data_dict[cur_date][2] += cur_deaths
            all_date.append(cur_date)
    for day in all_date:
        result.append((day, data_dict[day][0], data_dict[day][1], data_dict[day][2]))
    return np.array(result)



if __name__ == '__main__':
    f = open('covid-19_data.json')
    real_data = json.load(f)
