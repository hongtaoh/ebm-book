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

def process_chen_data(file):
    """Prepare data for analysis below
    """
    df = pd.read_excel(file)
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
    num_participant = len(df.participant.unique())
    participant_string_id_dic = dict(zip(df.participant.unique(), [_ for _ in range(num_participant)]))
    df['participant'] = df.apply(lambda row: participant_string_id_dic[row.participant], axis = 1 )
    return df 

def get_data_we_have(data_source):
    if data_source == "Chen Data":
         data_we_have = process_chen_data("data/Chen2016Data.xlsx")
    else:
        original_data = pd.read_csv('data/participant_data.csv')
        original_data['diseased'] = original_data.apply(lambda row: row.k_j > 0, axis = 1)
        data_we_have = original_data.drop(['k_j', 'S_n', 'affected_or_not'], axis = 1)
    return data_we_have

def run_conjugate_priors(
        data_source,
        log_folder_name, 
        img_folder_name
    ):
        
    data_we_have = get_data_we_have(data_source)
    biomarkers = data_we_have.biomarker.unique()
    theta_phi_kmeans = utils.get_theta_phi_kmeans(data_we_have, biomarkers, n_clusters = 2)

    print(f"Now begins with {data_source} with conjugate priors")
    start_time = time.time()
    biomarker_best_order_dic, \
    participant_stages, \
    all_dicts, \
    all_current_participant_stages,\
    all_current_best_order_dicts, \
    all_current_best_likelihoods, \
    all_current_acceptance_ratios, \
    final_acceptance_ratio = utils.metropolis_hastings_with_conjugate_priors(
        data_we_have, iterations, theta_phi_kmeans, log_folder_name
    )
    utils.save_heatmap(
        all_dicts, burn_in, thining, 
        folder_name=img_folder_name,
        file_name="heatmap_all_orderings",
        title = f"{data_source} with Conjugate Priors, All Orderings"
    )
    utils.save_heatmap(
        all_current_best_order_dicts, 
        burn_in=0, thining=1, 
        folder_name=img_folder_name,
        file_name = "heatmap_all_current_best",
        title = f"{data_source} with Conjugate Priors, All Current Best Orderings"
    )
    utils.save_trace_plot(
        all_current_best_likelihoods, 
        folder_name=img_folder_name,
        file_name="trace_plot",
        title = f"Trace Plot, {data_source} with Conjugate Priors"
    )
    end_time = time.time()
    execution_time = (end_time - start_time)/60
    print(f"Execution time: {execution_time} min for {data_source} using conjugate priors.")
    print("---------------------------------------------")

def run_kmeans_only(
        data_source,
        log_folder_name, 
        img_folder_name
    ):
    data_we_have = get_data_we_have(data_source)
    biomarkers = data_we_have.biomarker.unique()
    theta_phi_kmeans = utils.get_theta_phi_kmeans(data_we_have, biomarkers, n_clusters = 2)

    print(f"Now begins with {data_source} with kmeans only")
    start_time = time.time()
    biomarker_best_order_dic, \
    all_dicts, \
    all_current_best_order_dicts, \
    all_current_best_likelihoods, \
    all_current_acceptance_ratios, \
    final_acceptance_ratio = utils.metropolis_hastings_theta_phi_kmeans_and_average_likelihood(
        data_we_have, iterations, theta_phi_kmeans, log_folder_name
    )

    utils.save_heatmap(
        all_dicts, burn_in, thining, 
        folder_name=img_folder_name,
        file_name = "heatmap_all_orderings",
        title = f"{data_source} with KMeans Only, All Orderings"
    )
    utils.save_heatmap(
        all_current_best_order_dicts, 
        burn_in=0, thining=1, 
        folder_name=img_folder_name,
        file_name = "heatmap_all_current_best",
        title = f"{data_source} with KMeans Only, All Current Best Orderings"
    )
    utils.save_trace_plot(
        all_current_best_likelihoods, 
        folder_name=img_folder_name, 
        file_name="trace_plot",
        title = f"Trace Plot, {data_source} with KMeans Only"
    )
    end_time = time.time()
    execution_time = (end_time - start_time)/60
    print(f"Execution time: {execution_time} mins for {data_source} using kmeans only.")
    print("---------------------------------------------")

if __name__ == '__main__':
     
    iterations = 1000
    burn_in = 500
    thining = 50

    # """Simulated Data
    # """
    # # Simulated data with conjugate priors
    # run_conjugate_priors(
    #      data_source = "Simulated Data",
    #      log_folder_name = "logs/simulated_data_conjugate_priors",
    #      img_folder_name = "img/simulated_data_conjugate_priors"
    # )
    # # Simulated data with kmeans only
    # run_kmeans_only(
    #     data_source = "Simulated Data",
    #     log_folder_name = "logs/simulated_data_kmeans_only", 
    #     img_folder_name = "img/simulated_data_kmeans_only"
    # )

    # """Chen Data
    # """
    # # Chen data with conjugate priors
    # run_conjugate_priors(
    #      data_source = "Chen Data",
    #      log_folder_name = "logs/chen_data_conjugate_priors", 
    #      img_folder_name = "img/chen_data_conjugate_priors"
    # )
    # Chen data with kmeans only
    run_kmeans_only(
         data_source = "Chen Data",
         log_folder_name = "logs/chen_data_kmeans_only", 
        img_folder_name = "img/chen_data_kmeans_only"
    )
    
    
   











