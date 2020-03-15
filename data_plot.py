########################################################################################################################
#                                                                                                                      #
#                                            Covid-19-metrics: plot module                                             #
#                                                                                                                      #
#                                               Lionel Cheng 15.03.2020                                                #
#                                                                                                                      #
########################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from data_processing import read_data, log_reglin, logistic_model, retrieve_region, evolution
import matplotlib as mpl

mpl.rcParams['lines.linewidth'] = 2.5
mpl.rcParams['lines.markersize'] = 7

def plot_model(days, data, models, model_names, model_days):
    # plot
    fig, axes = plt.subplots(ncols=2, figsize=(12, 6))
    for model, model_name, model_day in zip(models, model_names, model_days):
        axes[0].plot(model_day, model, label=model_name)
    axes[0].plot(days, data, label='Real data')
    axes[0].legend()
    axes[0].set_ylim(-0.05 * max(data), 1.1 * max(data))
    axes[0].grid(True)
    for model, model_name, model_day in zip(models, model_names, model_days):
        axes[1].plot(model_day, model, label=model_name)
    axes[1].plot(days, data, label='Real data')
    axes[1].legend()
    axes[1].set_yscale('log')
    axes[1].grid(True)

def plot_ax(ax, x, y, label=None, scale='linear', title=None):
    if label is not None:
        ax.plot(x, y, label=label)
        ax.legend()
    else:
        ax.plot(x, y)
    ax.set_yscale(scale)
    ax.grid(True)
    if title is not None:
        ax.set_title(title)

def print_cases(region, conf, death, recov):
    print('%s - Cases : %d - Deaths : %d - Recovered : %d ' % (region, conf[-1], death[-1], recov[-1]))

def region_plot(country, cases, death, recov, days, country_tot, stop_exp=0, exclude=False):
    ndays = len(days)
    # model of exponential growth and plot in a country
    total_conf_country = retrieve_region(country, cases, ndays, exclude=exclude)
    total_death_country = retrieve_region(country, death, ndays, exclude=exclude)
    total_recov_country = retrieve_region(country, recov, ndays, exclude=exclude)

    if stop_exp != 0:
        country_model_1, R, a, b, alpha, model_1_days = log_reglin(days[:stop_exp], total_conf_country[:stop_exp])
    else:
        country_model_1, R, a, b, alpha, model_1_days = log_reglin(days, total_conf_country)
    if exclude:
        print('World w/o %s - R = %.2f - a = %.2e - b = %.2e - alpha = %.2f' % (country, R, a, b, alpha))
    else:
        print('%s - R = %.2f - a = %.2e - b = %.2e - alpha = %.2f' % (country, R, a, b, alpha))
    country_model_2, model_2_days = logistic_model(days, alpha, country_tot, total_conf_country)

    models = [country_model_1, country_model_2]
    model_names = ['Exponential model', 'Logistic model']
    model_days = [model_1_days, model_2_days]

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 10))

    for model, model_name, model_day in zip(models, model_names, model_days):
        axes[0][0].plot(model_day, model, label=model_name)
    plot_ax(axes[0][0], days, total_conf_country, label='Real data', title='Linear scale cases')

    for model, model_name, model_day in zip(models, model_names, model_days):
        axes[0][1].plot(model_day, model, label=model_name)
    plot_ax(axes[0][1], days, total_conf_country, label='Real data', scale='log', title='Log scale cases')

    plot_ax(axes[0][2], days, total_death_country, title='Deaths')
    plot_ax(axes[0][3], days, total_recov_country, title='Recovered')

    increase, growth_factor = evolution(days, total_conf_country)

    plot_ax(axes[1][0], days, increase, title=r'Daily increase $\Delta N_d$')
    plot_ax(axes[1][1], days, growth_factor, title=r'Growth factor $\Delta N_d/\Delta N_{d-1}$')

    axes[1][2].plot(days, 100 * total_death_country / total_conf_country, label= r'Inf $d_r$')
    plot_ax(axes[1][2], days, 100 * total_death_country / total_recov_country, label=r'Sup $d_r$', title='Death rate')
    axes[1][2].set_ylim(0, 10)
    axes[1][2].set_ylabel('Percentage')

    doubling_period = np.log(2) / a

    if exclude:
        plt.suptitle('World w/o ' + country + " - Start date : 22nd January \n " + 
            r"Exponential model: $\alpha$ = %.2f - Doubling period : %.2f days" % (alpha, doubling_period) + "\n"
            r"Logistic model: $n_{tot}$ = %d - $\alpha$ = %.2f" % (country_tot, alpha))
        plt.savefig('FIGURES/world_wo_%s_data' % country, bbox_inches='tight')
    else:
        plt.suptitle(country + " - Start date : 22nd January \n " + 
            r"Exponential model: $\alpha$ = %.2f - Doubling period : %.2f days" % (alpha, doubling_period) + "\n"
            r"Logistic model: $n_{tot}$ = %d - $\alpha$ = %.2f" % (country_tot, alpha))
        plt.savefig('FIGURES/%s_data' % country, bbox_inches='tight')

def plot_ax_global(ax, days, total, total_china, total_out_china, title):
    ax.plot(days, total, label='Worldwide')
    ax.plot(days, total_china, label='China')
    ax.plot(days, total_out_china, label='Outside China')
    ax.legend()
    ax.grid(True)
    ax.set_title(title)

def plot_ax_deathrate(ax, days, conf, death, recov, title):
    ax.plot(days, 100 * death / recov, label=r'Sup $d_r$')
    ax.plot(days, 100 * death / conf, label=r'Inf $d_r$')
    ax.legend()
    ax.set_title(title)
    ax.set_ylim(0, 10)
    ax.set_ylabel('Percentage')
    ax.grid(True)
