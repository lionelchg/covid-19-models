########################################################################################################################
#                                                                                                                      #
#                                               Covid-19-metrics: main                                                 #
#                                                                                                                      #
#                                               Lionel Cheng 15.03.2020                                                #
#                                                                                                                      #
########################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from data_processing import read_data, log_reglin, logistic_model, retrieve_region, evolution
from data_plot import plot_ax_global, plot_model, plot_ax, print_cases, region_plot, plot_ax_deathrate

if __name__ == '__main__':
    directory = 'COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-'
    files = ['Confirmed', 'Deaths', 'Recovered']
    conf, total_conf, total_conf_china, total_conf_out_china = read_data(directory + files[0] + '.csv')
    death, total_death, total_death_china, total_death_out_china = read_data(directory + files[1] + '.csv')
    recov, total_recov, total_recov_china, total_recov_out_china = read_data(directory + files[2] + '.csv')

    regions_count = len(conf)

    ndays = len(conf[0][1])
    days = np.arange(ndays, dtype=np.float64)

    print_cases('World', total_conf, total_death, total_recov)
    print_cases('China', total_conf_china, total_death_china, total_recov_china)
    print_cases('Rest of the world', total_conf_out_china, total_death_out_china, total_recov_out_china)

    region_plot('France', conf, death, recov, days, 3e5)
    region_plot('Italy', conf, death, recov, days, 6e5)
    region_plot('US', conf, death, recov, days, 1e6)
    region_plot('China', conf, death, recov, days, 8.2e4, stop_exp=20)
    region_plot('China', conf, death, recov, days, 1e6, exclude=True)

    fig, axes = plt.subplots(ncols=3, figsize=(14, 5), sharey=True)
    plot_ax_deathrate(axes[0], days,total_conf, total_death, total_recov, 'World')
    plot_ax_deathrate(axes[1], days,total_conf_china, total_death_china, total_recov_china, 'China')
    plot_ax_deathrate(axes[2], days,total_conf_out_china, total_death_out_china, total_recov_out_china, 'World w/o China')

    plt.savefig('FIGURES/GLOBAL/world_death_rate', bbox_inches='tight')

    fig, axes = plt.subplots(ncols=3, figsize=(14, 5))
    plot_ax_global(axes[0], days, total_conf, total_conf_china, total_conf_out_china, files[0])
    plot_ax_global(axes[1], days, total_death, total_death_china, total_death_out_china, files[1])
    plot_ax_global(axes[2], days, total_recov, total_recov_china, total_recov_out_china, files[2])

    plt.savefig('FIGURES/GLOBAL/world_cases', bbox_inches='tight')