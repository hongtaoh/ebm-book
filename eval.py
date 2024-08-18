import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
from scipy.stats import norm
from scipy.stats import mode
from sklearn.cluster import KMeans
from matplotlib.animation import FuncAnimation
import time
import random
import utils

def process_chen_data(file):
    """Prepare data for analysis below
    """
    df = pd.read_excel(file)
    biomarker_name_change_dic = dict(zip(['FCI(HIP)', 'GMI(HIP)', 'FCI(Fusi)', 'FCI(PCC)', 'GMI(FUS)'],
                                         [1, 3, 5, 2, 4]))
    df.rename(
        columns={df.columns[0]: 
                 'participant_category', df.columns[1]: 
                 'participant'}, 
                 inplace=True)
    # df = df[df.participant_category.isin(['CN', 'AD '])]
    df['diseased'] = df.apply(lambda row: row.participant_category != 'CN', axis = 1)
    df = pd.melt(df, id_vars=['participant_category', "participant", "timestamp", 'diseased'], 
                        value_vars=["FCI(HIP)", "GMI(HIP)", "FCI(Fusi)", "FCI(PCC)", "GMI(FUS)"], 
                        var_name='biomarker', value_name='measurement')
    # convert participant id
    n_participant = len(df.participant.unique())
    participant_ids = [_ for _ in range(n_participant)]
    participant_string_id_dic = dict(zip(df.participant.unique(), participant_ids))
    df['participant'] = df.apply(lambda row: participant_string_id_dic[row.participant], axis = 1 )
    df['biomarker'] = df.apply(lambda row: f"{row.biomarker}-{biomarker_name_change_dic[row.biomarker]}", 
                               axis = 1)
    return df 

def get_data_we_have(data_source):
    if data_source == "Chen Data":
         data_we_have = process_chen_data("data/Chen2016Data.xlsx")
    else:
        biomarker_name_change_dic = dict(zip(['HIP-FCI', 'HIP-GMI', 'FUS-FCI', 'PCC-FCI', 'FUS-GMI'],
                                         [1, 3, 5, 2, 4]))
        original_data = pd.read_csv('data/participant_data.csv')
        original_data['diseased'] = original_data.apply(lambda row: row.k_j > 0, axis = 1)
        data_we_have = original_data.drop(['k_j', 'S_n', 'affected_or_not'], axis = 1)
        data_we_have['biomarker'] = data_we_have.apply(
            lambda row: f"{row.biomarker} ({biomarker_name_change_dic[row.biomarker]})", axis = 1)
    return data_we_have

def run_conjugate_priors(
        data_source,
        iterations,
        log_folder_name, 
        img_folder_name,
        n_shuffle
    ):
        
    data_we_have = get_data_we_have(data_source)

    print(f"Now begins with {data_source} with conjugate priors")
    start_time = time.time()
    biomarker_best_order_dic, \
    participant_stages, \
    all_dicts, \
    all_current_participant_stages,\
    all_current_order_dicts, \
    all_current_likelihoods, \
    all_current_acceptance_ratios, \
    final_acceptance_ratio = utils.metropolis_hastings_with_conjugate_priors(
        data_we_have, iterations, log_folder_name, n_shuffle
    )
    utils.save_heatmap(
        all_dicts, burn_in, thining, 
        folder_name=img_folder_name,
        file_name="heatmap_all_orderings",
        title = f"{data_source} with Conjugate Priors, All Orderings"
    )
    utils.save_heatmap(
        all_current_order_dicts, 
        burn_in=0, thining=1, 
        folder_name=img_folder_name,
        file_name = "heatmap_all_current",
        title = f"{data_source} with Conjugate Priors, All Current Best Orderings"
    )
    utils.save_trace_plot(
        burn_in,
        all_current_likelihoods, 
        folder_name=img_folder_name,
        file_name="trace_plot",
        title = f"Trace Plot, {data_source} with Conjugate Priors"
    )
    end_time = time.time()
    execution_time = (end_time - start_time)/60
    print(f"Execution time: {execution_time} min for {data_source} using conjugate priors.")
    print("---------------------------------------------")

def run_soft_kmeans(
        data_source,
        iterations,
        log_folder_name, 
        img_folder_name
    ):
    data_we_have = get_data_we_have(data_source)
    # theta_phi_estimates = pd.read_csv('data/means_stds.csv')

    print(f"Now begins with {data_source} with soft kmeans")
    start_time = time.time()
    current_accepted_order_dict, \
    all_order_dicts, \
    all_current_accepted_order_dicts, \
    all_current_accepted_likelihoods,\
    all_current_acceptance_ratios, \
    final_acceptance_ratio = utils.metropolis_hastings_soft_kmeans(
        data_we_have, iterations, log_folder_name
    )

    utils.save_heatmap(
        all_order_dicts, burn_in, thining, 
        folder_name=img_folder_name,
        file_name = "heatmap_all_orderings",
        title = f"{data_source} with Soft KMeans, All Orderings"
    )
    utils.save_heatmap(
        all_current_accepted_order_dicts, 
        burn_in=0, thining=1, 
        folder_name=img_folder_name,
        file_name = "heatmap_all_current_accepted",
        title = f"{data_source} with Soft KMeans, All Current Accepted Orderings"
    )
    utils.save_trace_plot(
        burn_in,
        all_current_accepted_likelihoods, 
        folder_name=img_folder_name, 
        file_name="trace_plot",
        title = f"Trace Plot, {data_source} with Soft KMeans"
    )
    end_time = time.time()
    execution_time = (end_time - start_time)/60
    print(f"Execution time: {execution_time} mins for {data_source} using soft kmeans.")
    print("---------------------------------------------")

def run_kmeans(
        data_source,
        iterations,
        log_folder_name, 
        img_folder_name,
        real_order,
        burn_in, 
        thining
    ):
    data_we_have = get_data_we_have(data_source)
    # theta_phi_estimates = pd.read_csv('data/means_stds.csv')

    print(f"Now begins with {data_source} with kmeans")
    start_time = time.time()
    current_accepted_order_dict, \
    all_order_dicts, \
    all_current_accepted_order_dicts, \
    all_current_accepted_likelihoods,\
    all_current_acceptance_ratios, \
    final_acceptance_ratio = utils.metropolis_hastings_kmeans(
        data_we_have, iterations, log_folder_name, real_order, burn_in, thining
    )

    utils.save_heatmap(
        all_order_dicts, burn_in, thining, 
        folder_name=img_folder_name,
        file_name = "heatmap_all_orderings",
        title = f"{data_source} with KMeans, All Orderings"
    )
    utils.save_heatmap(
        all_current_accepted_order_dicts, 
        burn_in=0, thining=1, 
        folder_name=img_folder_name,
        file_name = "heatmap_all_current_accepted",
        title = f"{data_source} with KMeans, All Current Accepted Orderings"
    )
    utils.save_trace_plot(
        burn_in,
        all_current_accepted_likelihoods, 
        folder_name=img_folder_name, 
        file_name="trace_plot",
        title = f"Trace Plot, {data_source} with KMeans"
    )
    end_time = time.time()
    execution_time = (end_time - start_time)/60
    print(f"Execution time: {execution_time} mins for {data_source} using kmeans.")
    print("---------------------------------------------")

if __name__ == '__main__':
     
    iterations = 100
    burn_in = 10
    thining = 2
    n_shuffle = 2
    real_order = [1, 3, 5, 2, 4]

    # """Simulated Data
    # """
    # # Simulated data with conjugate priors
    # run_conjugate_priors(
    #      data_source = "Simulated Data",
    #      iterations=iterations,
    #      log_folder_name = "logs/simulated_data_conjugate_priors",
    #      img_folder_name = "img/simulated_data_conjugate_priors",
    #      n_shuffle = n_shuffle
    # )
    # # Simulated data with kmeans
    # run_soft_kmeans(
    #     data_source = "Simulated Data",
    #     iterations=iterations,
    #     log_folder_name = "logs/simulated_data_soft_kmeans", 
    #     img_folder_name = "img/simulated_data_soft_kmeans"
    # )
    # Soley kmeans
    run_kmeans(
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
    # run_conjugate_priors(
    #     data_source = "Chen Data",
    #     iterations=iterations,
    #     log_folder_name = "logs/chen_data_conjugate_priors", 
    #     img_folder_name = "img/chen_data_conjugate_priors",
    #     n_shuffle=n_shuffle
    # )
    # Chen data with soft kmeans
    # run_soft_kmeans(
    #     data_source = "Chen Data",
    #     iterations=iterations,
    #     log_folder_name = "logs/chen_data_soft_kmeans", 
    #     img_folder_name = "img/chen_data_soft_kmeans"
    # )
    run_kmeans(
        data_source = "Chen Data",
        iterations=iterations,
        log_folder_name = "logs/chen_data_kmeans", 
        img_folder_name = "img/chen_data_kmeans",
        real_order = real_order,
        burn_in = burn_in, 
        thining = thining
    )
    
    
   











