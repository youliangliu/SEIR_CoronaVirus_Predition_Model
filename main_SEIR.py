import math
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import matplotlib.widgets  # Cursor
import matplotlib.dates
import matplotlib.ticker

import shared

import helper

COUNTRY = 'Germany'  # e.g. 'all' South Korea' 'France'  'Republic of Korea' 'Italy' 'Germany'  'US' 'Spain'
PROVINCE = 'all'  # 'all' 'Hubei'  # for provinces other than Hubei the population value needs to be set manually
EXCLUDECOUNTRIES = ['China'] if COUNTRY == 'all' else []  # massive measures early on

population_dict, actual_data = helper.pre_processing()

population = int(population_dict[COUNTRY])

#population = population.get_population(COUNTRY, PROVINCE, EXCLUDECOUNTRIES)

# --- parameters ---


import scipy.integrate
import matplotlib.pyplot as plt

total_days = 565
lockdown_day = 70

def R_0(t):
    return 2.4 if t < lockdown_day else 0.85

N = population
time_presymptom = 2.0
delta = 1.0 / (5.2 - time_presymptom)
# https://www.medrxiv.org/content/10.1101/2020.03.05.20031815v1
# http://www.cidrap.umn.edu/news-perspective/2020/03/short-time-between-serial-covid-19-cases-may-hinder-containment
generation_time = 4.6
# for SEIR: generationTime = 1/delta + 0.5 * 1/gamma = timeFromInfectionToInfectiousness + timeInfectious
# gamma = 1/(2 * (generation_time - 1/delta))
# https://en.wikipedia.org/wiki/Serial_interval
gamma = 1.0 / (2.0 * (generation_time - 1.0 / delta))

def beta(t):
    return R_0(t) * gamma

###
# Without the Dead-Compartment
###
def deriv(y, t, N, beta, gamma, delta):
    S, E, I, R = y

    dSdt = -beta(t) * S * I / N
    dEdt = beta(t) * S * I / N - delta * E
    dIdt = delta * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt

S0, E0, I0, R0 = N-1, 1, 0, 0  # initial conditions: one exposed

t = np.arange(total_days)  # time steps array
y0 = S0, E0, I0, R0 # Initial conditions vector

# Integrate the SIR equations over the time grid, t.
ret = scipy.integrate.odeint(deriv, y0, t, args=(N, beta, gamma, delta))
S, E, I, R = ret.T

ICUs = 20000

no_symptoms = 0.35  # https://www.zmescience.com/medicine/iceland-testing-covid-19-0523/  but virus can already be found in throat 2.5 days before symptoms (Drosten)
find_ratio = (1.0 - no_symptoms) / 17.0  # !!! wild guess! a lot of the mild cases will go undetected  assuming 100% correct tests

time_in_hospital = 12
time_infected = 1.0 / gamma  # better timeInfectious?

# lag, whole days - need sources
presymptom_lag = round(time_presymptom)  # effort probably not worth to be more precise than 1 day
communication_lag = 2
test_lag = 3
symptom_to_hospital_lag = 5
hospital_to_ICU_lag = 5

alphaA = 0.005  # Diamond Princess age corrected, Heinsberg
alphaB = alphaA * 3.0  # higher lethality without ICU - by how much?  even higher without oxygen and meds
ICU_rate = alphaA * 2  # Imperial College NPI study: hospitalized/ICU/fatal = 6/2/1

# derived arrays
F = I * find_ratio
U = I * ICU_rate * time_in_hospital / time_infected
P = I / population * 1000000 # probability of random person to be infected# scale for short infectious time vs. real time in hospital

# timeline: exposed, infectious, symptoms, at home, hospital, ICU
F = shared.delay(F, presymptom_lag + symptom_to_hospital_lag + test_lag + communication_lag)  # found in tests and officially announced; from I
U = shared.delay(U, presymptom_lag + symptom_to_hospital_lag + hospital_to_ICU_lag)  # ICU  from I before delay

FC = np.cumsum(F)

D = np.arange(total_days)
R_prev = 0
D_prev = 0
for i, tt in enumerate(t):
    IFR = alphaA if U[i] <= ICUs else alphaB
    D[i] = D_prev + IFR * (R[i] - R_prev)
    R_prev = R[i]
    D_prev = D[i]

D = shared.delay(D, -time_infected + time_presymptom +symptom_to_hospital_lag + time_in_hospital + communication_lag)  # deaths  from R




# RPrev = 0
# DPrev = 0
# for i, x in enumerate(X):
#     IFR = infectionFatalityRateA if U[i] <= intensiveUnits else infectionFatalityRateB
#     D[i] = DPrev + IFR * (R[i] - RPrev)
#     RPrev = R[i]
#     DPrev = D[i]
#
# D = helper.delay(D, - timeInfected + timePresymptomatic +symptomToHospitalLag + timeInHospital + communicationLag)  # deaths  from R

# Plot
fig = plt.figure(dpi=75, figsize=(20,16))
ax = fig.add_subplot(111)
ax.fmt_xdata = matplotlib.dates.DateFormatter('%Y-%m-%d')  # higher date precision for cursor display

ax.set_yscale("log", nonposy='clip')

# actual country data
XCDR_data = helper.get_country_xcdr(COUNTRY, actual_data)
#dataOffset = shared.get_offset_X(XCDR_data, D, dataOffset)  # match model day to real data day for deaths curve  todo: percentage wise?
dataOffset = 149
ax.plot(XCDR_data[:,0], XCDR_data[:,1], 'o', color='orange', alpha=0.5, lw=1, label='cases actually detected in tests')
ax.plot(XCDR_data[:,0], XCDR_data[:,2], 'x', color='black', alpha=0.5, lw=1, label='actually deceased')

# set model time to real world time
X = np.arange(total_days)
X = shared.model_to_world_time(X - dataOffset, XCDR_data)

# model data
#ax.plot(X, S, 'b', alpha=0.5, lw=2, label='Susceptible')
ax.plot(X, E, 'y', alpha=0.5, lw=2, label='Exposed (realtime)')
ax.plot(X, I, 'r--', alpha=0.5, lw=1, label='Infected (realtime)')
ax.plot(X, FC, color='orange', alpha=0.5, lw=1, label='Found cumulated: "cases" (lagtime)')
ax.plot(X, U, 'r', alpha=0.5, lw=2, label='ICU (realtime)')
#ax.plot(X, R, 'g', alpha=0.5, lw=1, label='Recovered with immunity')
#ax.plot(X, P, 'c', alpha=0.5, lw=1, label='Probability of infection')
ax.plot(X, D, 'k', alpha=0.5, lw=1, label='Deaths (lagtime)')

# ax.plot([min(X), max(X)], [intensiveUnits, intensiveUnits], 'b-.', alpha=0.5, lw=1, label='Number of ICU available')

ax.set_xlabel('Time /days')
ax.set_ylim(bottom=1.0)
ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
ax.xaxis.set_minor_locator(matplotlib.dates.WeekdayLocator())
ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=100, base=10.0, subs=(1.0,)))
ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(numticks=100, base=10.0,
                                                        subs=np.arange(2, 10) * 0.1))


ax.grid(linestyle=':')  #b=True, which='major', c='w', lw=2, ls='-')
# if EXCLUDECOUNTRIES:
#     locationString = COUNTRY + "but " + ",".join(EXCLUDECOUNTRIES) + ' %dk' % (population / 1000)
# locationString = COUNTRY + " " + PROVINCE +' %dk' % (population / 1000)
# icuString = "  intensive care units: %.0f" % intensiveUnits + " (guess)"
# legend = ax.legend(title='COVID-19 SEIR model: ' + locationString + ' (beta)\n' + icuString)
# legend.get_frame().set_alpha(0.5)
# for spine in ('top', 'right', 'bottom', 'left'):
#     ax.spines[spine].set_visible(False)
# cursor = matplotlib.widgets.Cursor(ax, color='black', linewidth=1 )
#
# # text output
# print("sigma: %.3f  1/sigma: %.3f    gamma: %.3f  1/gamma: %.3f" % (sigma, 1.0/sigma, gamma, 1.0/gamma))
# print("beta0: %.3f" % beta0, "   beta1: %.3f" % beta1)

def print_info(i):
    print("day %d" % i)
    print(" Infected: %d" % I[i], "%.1f" % (I[i] * 100.0 / population))
    print(" Infected found: %d" % F[i], "%.1f" % (F[i] * 100.0 / population))
    print(" Infected found cumulated ('cases'): %d" % FC[i], "%.1f" % (FC[i] * 100.0 / population))
    print(" Hospital: %d" % U[i], "%.1f" % (U[i] * 100.0 / population))
    print(" Recovered: %d" % R[i], "%.1f" % (R[i] * 100.0 / population))
    print(" Deaths: %d" % D[i], "%.1f" % (D[i] * 100.0 / population))

print_info(lockdown_day)
print_info(total_days - 1)
# print("findratio: %.1f%%" % (findRatio * 100.0))
# print("doubling0 every ~%.1f" % doublingTime, "days")
# print("lockdown measures start:", X[days0])

if 1:
    plt.show()
else:
    plt.savefig('model_run.png')
