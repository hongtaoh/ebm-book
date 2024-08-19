import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
from scipy.stats import mode
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import random 
import math
import json
import time 
import os 
import utils

if __name__ == '__main__':
     
    iterations = 100
    burn_in = 10
    thining = 2
    n_shuffle = 2
    real_order = [1, 3, 5, 2, 4]
    S_ordering = np.array([
        'HIP-FCI', 'PCC-FCI', 'HIP-GMI', 'FUS-GMI', 'FUS-FCI'
    ])
    real_theta_phi = pd.read_csv('data/means_stds.csv')

    ns = [25, 50, 100, 200]
    rs = [0.1, 0.2, 0.4, 0.5]

    def task_parallelism()

    """Simulated Data
    """
    # Simulated data with conjugate priors
    utils.run_conjugate_priors(
        data_source = "Simulated Data",
        iterations=iterations,
        log_folder_name = "logs/simulated_data_conjugate_priors",
        img_folder_name = "img/simulated_data_conjugate_priors",
        n_shuffle = n_shuffle,
        burn_in = burn_in, 
        thining = thining
    )
    # Simulated data with kmeans
    utils.run_soft_kmeans(
        data_source = "Simulated Data",
        iterations=iterations,
        log_folder_name = "logs/simulated_data_soft_kmeans", 
        img_folder_name = "img/simulated_data_soft_kmeans",
        burn_in = burn_in, 
        thining = thining
    )
    # Soley kmeans
    utils.run_kmeans(
        data_source = "Simulated Data",
        iterations=iterations,
        log_folder_name = "logs/simulated_data_kmeans", 
        img_folder_name = "img/simulated_data_kmeans",
        real_order = real_order,
        burn_in = burn_in, 
        thining = thining
    )
    # """Chen Data
    # """
    # # Chen data with conjugate priors
    # utils.run_conjugate_priors(
    #     data_source = "Chen Data",
    #     iterations=iterations,
    #     log_folder_name = "logs/chen_data_conjugate_priors", 
    #     img_folder_name = "img/chen_data_conjugate_priors",
    #     n_shuffle=n_shuffle,
    #     burn_in = burn_in, 
    #     thining = thining
    # )
    # # Chen data with soft kmeans
    # utils.run_soft_kmeans(
    #     data_source = "Chen Data",
    #     iterations=iterations,
    #     log_folder_name = "logs/chen_data_soft_kmeans", 
    #     img_folder_name = "img/chen_data_soft_kmeans",
    #     burn_in = burn_in, 
    #     thining = thining
    # )
    # utils.run_kmeans(
    #     data_source = "Chen Data",
    #     iterations=iterations,
    #     log_folder_name = "logs/chen_data_kmeans", 
    #     img_folder_name = "img/chen_data_kmeans",
    #     real_order = real_order,
    #     burn_in = burn_in, 
    #     thining = thining
    # )
    
    
   











