import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
from scipy.stats import norm
from scipy.stats import mode
from sklearn.cluster import KMeans
from matplotlib.animation import FuncAnimation
import time

import utils


if __name__ == '__main__':
    original_data = pd.read_csv('data/participant_data.csv')
    original_data['diseased'] = original_data.apply(lambda row: row.k_j > 0, axis = 1)
    data_we_have = original_data.drop(['k_j', 'S_n', 'affected_or_not'], axis = 1)
    theta_phi_kmeans = pd.read_csv("data/estimate_means_stds_kmeans.csv")
    biomarkers = data_we_have.biomarker.unique()
    num_biomarkers = len(biomarkers)

    iterations = 1000
    burn_in = 500
    thining = 10

    """theta_phi_means + average_likelihood
    """
    log_folder_name = "logs/kmeans_average_likelihood"
    biomarker_best_order_dic_kmeans_average_likelihood, \
    all_dicts_kmeans_average_likelihood, \
    all_current_best_order_dicts_kmeans_average_likelihood, \
    all_current_best_likelihoods_kmeans_average_likelihood, \
    all_current_acceptance_ratios_kmeans_average_likelihood, \
    final_acceptance_ratio_kmeans_average_likelihood = utils.metropolis_hastings_theta_phi_kmeans_and_average_likelihood(
        data_we_have, iterations, theta_phi_kmeans, log_folder_name
    )
    utils.save_heatmap(
        all_dicts_kmeans_average_likelihood, burn_in, thining, "heatmap_kmeans_average_likelihood"
    )
    utils.save_trace_plot(
        all_current_best_likelihoods_kmeans_average_likelihood, 
        "img", 
        "trace_plot_kmeans_average_likelihood"
    )

    """conjugate priors
    """
    log_folder_name = "logs/conjugate_priors"
    # conjugate_priors + theta_phi_means (as backup) + exact participant stages
    biomarker_best_order_dic_conjugate_priors, \
    participant_stages_conjugate_priors, \
    all_dicts_conjugate_priors, \
    all_current_best_order_dicts_conjugate_priors, \
    all_current_best_likelihoods_conjugate_priors, \
    all_current_acceptance_ratios_conjugate_priors, \
    final_acceptance_ratio_conjugate_priors = utils.metropolis_hastings_with_conjugate_priors(
        data_we_have, iterations, theta_phi_kmeans, log_folder_name
    )
    utils.save_heatmap(
        all_dicts_conjugate_priors, burn_in, thining, "heatmap_conjugate_priors"
    )
    utils.save_trace_plot(
        all_current_best_likelihoods_conjugate_priors, 
        "img", 
        "trace_plot_conjugate_priors"
    )
    
    # utils.save_plot(all_dicts_conjugate_priors, num_biomarkers, 'img/heatmap_conjugate_priors')
    # utils.save_trace_plot(all_current_best_likelihoods_conjugate_priors, iterations, "img/trace_plot_conjugate_priors")
    # print(all_current_best_likelihoods_conjugate_priors)
    # print(np.arange(start = 1, stop = iterations, step = 1))











