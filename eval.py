import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
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
import concurrent.futures

# def task_parallelism_simulated(
#         iterations, 
#         burn_in, 
#         thining, 
#         n_shuffle, 
#         real_order,
#         ns, 
#         rs,
#         participants_data,
#     ):
#     """Simulated Data"""
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         executor.submit(
#             utils.run_conjugate_priors,
#             participants_data,
#             iterations,
#             "logs/simulated_data_conjugate_priors",
#             "img/simulated_data_conjugate_priors",
#             n_shuffle,
#             burn_in, 
#             thining,
#         )
#         executor.submit(
#             utils.run_soft_kmeans,
#             participants_data,
#             iterations,
#             "logs/simulated_data_conjugate_priors",
#             "img/simulated_data_conjugate_priors",
#             n_shuffle,
#             burn_in, 
#             thining,
#         )
#         executor.submit(
#             utils.run_kmeans,
#             participants_data,
#             iterations,
#             "logs/simulated_data_kmeans", 
#             "img/simulated_data_kmeans",
#             real_order,
#             burn_in, 
#             thining
#         )

if __name__ == '__main__':
    iterations = 2000
    burn_in = 1000
    thining = 20
    n_shuffle = 2
    real_order = [1, 3, 5, 2, 4]
    S_ordering = np.array([
        'HIP-FCI', 'PCC-FCI', 'HIP-GMI', 'FUS-GMI', 'FUS-FCI'
    ])
    real_theta_phi = pd.read_csv('data/means_stds.csv')

    ns = [25, 50, 100, 150, 200]
    rs = [0.1, 0.2, 0.4, 0.5, 0.6]

    participants_data = utils.generate_data_from_ebm(
        n_participants = ns[-2], 
        S_ordering = S_ordering, 
        real_theta_phi = real_theta_phi, 
        healthy_ratio = rs[1],
        seed=1234,
    )

    # start_time = time.time()
    # task_parallelism_simulated(
    #     iterations, 
    #     burn_in, 
    #     thining, 
    #     n_shuffle, 
    #     real_order,
    #     ns, 
    #     rs,
    #     participants_data
    # )
    # task_parallelism_time = time.time() - start_time
    # print(f"Task Parallelism Only execution time: {task_parallelism_time:.4f} seconds")


    for uniform_prior in [True, False]:
        if uniform_prior:
            text = "uniform_prior"
        else:
            text = "non_uniform_prior"
        # Simulated data with conjugate priors
        utils.run_conjugate_priors(
            data_we_have = participants_data,
            iterations=iterations,
            log_folder_name = f"logs/{text}/simulated_data_conjugate_priors",
            img_folder_name = f"img/{text}/simulated_data_conjugate_priors",
            n_shuffle = n_shuffle,
            burn_in = burn_in, 
            thining = thining,
            uniform_prior=uniform_prior
        )
        # Simulated data with kmeans
        utils.run_soft_kmeans(
            data_we_have=participants_data,
            iterations=iterations,
            n_shuffle = n_shuffle,
            log_folder_name = f"logs/{text}/simulated_data_soft_kmeans", 
            img_folder_name = f"img/{text}/simulated_data_soft_kmeans",
            burn_in = burn_in, 
            thining = thining,
            uniform_prior=uniform_prior
        )
        # Soley kmeans
        utils.run_kmeans(
            data_we_have=participants_data,
            iterations=iterations,
            n_shuffle = n_shuffle,
            log_folder_name = f"logs/{text}/simulated_data_kmeans", 
            img_folder_name = f"img/{text}/simulated_data_kmeans",
            real_order = real_order,
            burn_in = burn_in, 
            thining = thining,
            uniform_prior=uniform_prior
        )
        
        """Chen Data
        """
        # Chen data with conjugate priors
        utils.run_conjugate_priors(
            data_we_have = participants_data,
            iterations=iterations,
            log_folder_name = f"logs/{text}/chen_data_conjugate_priors", 
            img_folder_name = f"img/{text}/chen_data_conjugate_priors",
            n_shuffle=n_shuffle,
            burn_in = burn_in, 
            thining = thining,
            chen_data=True,
            uniform_prior=uniform_prior
        )
        # Chen data with soft kmeans
        utils.run_soft_kmeans(
            data_we_have = participants_data,
            iterations=iterations,
            n_shuffle = n_shuffle,
            log_folder_name = f"logs/{text}/chen_data_soft_kmeans", 
            img_folder_name = f"img/{text}/chen_data_soft_kmeans",
            burn_in = burn_in, 
            thining = thining,
            chen_data=True,
            uniform_prior=uniform_prior
        )
        # Chen data with kmeans only
        utils.run_kmeans(
                data_we_have = participants_data,
                iterations=iterations,
                n_shuffle = n_shuffle,
                log_folder_name = f"logs/{text}/chen_data_kmeans", 
                img_folder_name = f"img/{text}/chen_data_kmeans",
                real_order = real_order,
                burn_in = burn_in, 
                thining = thining,
                chen_data=True,
                uniform_prior=uniform_prior
            )
    
   











