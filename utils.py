import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
from scipy.stats import mode
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import random 
import copy
import math

def get_theta_phi_for_single_biomarker(data, biomarker, clustering_setup):
    """To get theta and phi parametesr for a single biomarker 
    Input:
        - data: data we have right now, without access to S_n and kj
        - biomarker: a string of biomarker name
        - clustering_setup: kmeans_only, hierarchical_clustering, or both
    Output:
        mean and std of theta and phi
    """
    # two empty clusters to strore measurements
    clusters = [[] for _ in range(2)]

    # dataframe for this biomarker
    biomarker_df = data[data['biomarker'] == biomarker].reset_index(drop=True)
    # reshape to satisfy sklearn requirements
    measurements = np.array(biomarker_df['measurement']).reshape(-1, 1)

    # dataframe for non-diseased participants
    healthy_df = biomarker_df[biomarker_df['diseased'] == False]
    healthy_measurements = np.array(healthy_df['measurement']).reshape(-1, 1)

    # Fit clustering method
    clustering_result = clustering_setup.fit(measurements)

    if isinstance(clustering_setup, KMeans):
        predictions = clustering_result.predict(measurements)
    else:
        predictions = clustering_result.labels_
    
    # to store measurements into their cluster
    for i, prediction in enumerate(predictions):
        clusters[prediction].append(measurements[i][0])

    # which cluster are healthy participants in
    healthy_predictions = predictions[healthy_df.index]

    # the mode of the above predictions will be the phi cluster index
    phi_cluster_idx = mode(healthy_predictions, keepdims=False).mode
    theta_cluster_idx = 1 - phi_cluster_idx
    theta_mean, theta_std = np.mean(clusters[theta_cluster_idx]), np.std(clusters[theta_cluster_idx])
    phi_mean, phi_std = np.mean(clusters[phi_cluster_idx]), np.std(clusters[phi_cluster_idx])
    return theta_mean, theta_std, phi_mean, phi_std

def get_theta_phi_for_single_biomarker_using_kmeans_and_hierarchical_clustering(
        data, biomarker):
    """
    To get theta and phi parameters for a single biomarker using the K-means algorithm.
    Input:
        - data: DataFrame containing the data.
        - biomarker: A string representing the biomarker name.
        - kmeans_setup: An instance of KMeans from scikit-learn.
    Output:
        - Mean and standard deviation of theta and phi.
    """
    kmeans_setup = KMeans(n_clusters=2, random_state=0, n_init="auto")

    # two empty clusters to strore measurements
    clusters = [[] for _ in range(2)]

    # dataframe for this biomarker
    biomarker_df = data[data['biomarker'] == biomarker].reset_index(drop=True)
    # you need to make sure each measurment is a np.array before putting it into "fit"
    measurements = np.array(biomarker_df['measurement']).reshape(-1, 1)
    
    # dataframe for non-diseased participants
    healthy_df = biomarker_df[biomarker_df['diseased'] == False]
    healthy_measurements = np.array(healthy_df['measurement']).reshape(-1, 1)

    # Fit k-means
    kmeans = kmeans_setup.fit(measurements)
    predictions = kmeans.predict(measurements)

    # Verify that all healthy participants are in one cluster
    # which clusters are healthy participants in:
    healthy_predictions = kmeans.predict(healthy_measurements)
    # Identify healthy cluster index
    phi_cluster_idx = mode(healthy_predictions, keepdims=False).mode

    if len(set(healthy_predictions)) > 1:
        # Reassign clusters using Agglomerative Clustering
        clustering = AgglomerativeClustering(n_clusters=2).fit(healthy_measurements)

        # Find the dominant cluster for healthy participants
        phi_cluster_idx = mode(clustering.labels_, keepdims=False).mode 

         # Update predictions to ensure all healthy participants are in the dominant cluster
        updated_predictions = predictions.copy()
        for i in healthy_df.index:
            updated_predictions[i] = phi_cluster_idx
    else:
        updated_predictions = predictions 

    # Identify diseased cluster index
    theta_cluster_idx = 1 - phi_cluster_idx

    # Store measurements into their cluster
    for i, prediction in enumerate(updated_predictions):
        clusters[prediction].append(measurements[i][0])

    # Calculate means and standard deviations
    theta_mean, theta_std = np.mean(clusters[theta_cluster_idx]), np.std(clusters[theta_cluster_idx])
    phi_mean, phi_std = np.mean(clusters[phi_cluster_idx]), np.std(clusters[phi_cluster_idx])
    return theta_mean, theta_std, phi_mean, phi_std

def get_theta_phi_estimates(data_we_have, biomarkers, n_clusters, method):
    """
    Get the DataFrame of theta and phi using the K-means algorithm for all biomarkers.
    Input:
        - data_we_have: DataFrame containing the data.
        - biomarkers: List of biomarkers in string.
        - n_clusters: Number of clusters (should be 2).
    Output:
        - DataFrame containing the means and standard deviations for theta and phi for each biomarker.
    """
    kmeans_setup = KMeans(n_clusters, random_state=0, n_init="auto")
    hierarchical_clustering_setup = AgglomerativeClustering(n_clusters=2)
    # empty list of dictionaries to store the estimates 
    means_stds_estimate_dict_list = []
    for idx, biomarker in enumerate(biomarkers):
        dic = {'biomarker': biomarker}
        if method == "kmeans_only":
            theta_mean, theta_std, phi_mean, phi_std = get_theta_phi_for_single_biomarker(
                data_we_have, biomarker, kmeans_setup)
        elif method == "hierarchical_clustering_only":
            theta_mean, theta_std, phi_mean, phi_std = get_theta_phi_for_single_biomarker(
                data_we_have, biomarker, hierarchical_clustering_setup)
        elif method == "kmeans_and_hierarchical_clustering":
            theta_mean, theta_std, phi_mean, phi_std = get_theta_phi_for_single_biomarker_using_kmeans_and_hierarchical_clustering(
                data_we_have, biomarker)
        dic['theta_mean'] = theta_mean
        dic['theta_std'] = theta_std
        dic['phi_mean'] = phi_mean
        dic['phi_std'] = phi_std
        means_stds_estimate_dict_list.append(dic)
    return pd.DataFrame(means_stds_estimate_dict_list)

def fill_up_pdata(pdata, k_j):
    '''Fill up a single participant's data using k_j; basically add two columns: 
    k_j and affected
    Note that this function assumes that pdata already has the S_n column
    
    Input:
    - pdata: a dataframe of ten biomarker values for a specific participant 
    - k_j: a scalar
    '''
    data = pdata.copy()
    data['k_j'] = k_j
    data['affected'] = data.apply(lambda row: row.k_j >= row.S_n, axis = 1)
    return data 

def compute_single_measurement_likelihood(theta_phi, biomarker, affected, measurement):
    '''Computes the likelihood of the measurement value of a single biomarker
    We know the normal distribution defined by either theta or phi
    and we know the measurement. This will give us the probability
    of this given measurement value. 

    input:
    - theta_phi: the dataframe containing theta and phi values for each biomarker
    - biomarker: an integer between 0 and 9 
    - affected: boolean 
    - measurement: the observed value for a biomarker in a specific participant

    output: a scalar
    '''
    biomarker_params = theta_phi[theta_phi.biomarker == biomarker].reset_index()
    mu = biomarker_params['theta_mean'][0] if affected else biomarker_params['phi_mean'][0]
    std = biomarker_params['theta_std'][0] if affected else biomarker_params['phi_std'][0]
    var = std**2
    likelihood = np.exp(-(measurement - mu)**2/(2*var))/np.sqrt(2*np.pi*var)
    return likelihood

def compute_likelihood(pdata, k_j, theta_phi):
    '''This implementes the formula of https://ebm-book2.vercel.app/distributions.html#known-k-j
    This function computes the likelihood of seeing this sequence of biomarker values 
    for a specific participant, assuming that this participant is at stage k_j
    '''
    data = fill_up_pdata(pdata, k_j)
    likelihood = 1
    for i, row in data.iterrows():
        biomarker = row['biomarker']
        measurement = row['measurement']
        affected = row['affected']
        likelihood *= compute_single_measurement_likelihood(
            theta_phi, biomarker, affected, measurement)
    return likelihood

# def average_all_likelihood(pdata, num_biomarkers, theta_phi):
#     '''This is to compute https://ebm-book2.vercel.app/distributions.html#unknown-k-j
#     '''
#     return np.mean([compute_likelihood(
#         pdata=pdata, k_j=x, theta_phi=theta_phi) for x in range(
#             num_biomarkers+1)])

# def margnalized_likelihood
# cllapse versus non-collaps gibbs sampling in topic modeling

def weighted_average_likelihood(pdata, diseased_stages, normalized_stage_likelihood_dict, theta_phi):
    """using weighted average likelihood
    https://ebm-book2.vercel.app/distributions.html#unknown-k-j
    just that we do not assume each stage having exactly the same likelihood
    """
    weighted_average_ll = 0
    for x in diseased_stages:
        # likelihood: the product of the normalized likelihood of this participant being at this stage
        # and the likelihood of this participant having this sequence of biomarker measurements
        # assuming this participant is at this stage. 
        weighted_average_ll += normalized_stage_likelihood_dict[x] * compute_likelihood(pdata, x, theta_phi)
    return weighted_average_ll

# def compute_ln_likelihood_assuming_ordering(ordering_dic, data, num_biomarkers, theta_phi):
#     """Compute the (ln version of) the likelihood of seeing all participants' data,
#     assuming that we already know the ordering
#     Inputs:
#         - ordering: an array of ordering for biomarker 0-9
#         - data: data_we_have
#         - num_participants
#         - num_biomarkers 
#     Outputs:
#         - ln(likelihood)
#     """
#     num_participants = len(data.participant.unique())
#     # fill up S_n column using the ordering dict
#     # copy first in order not to change data_we_have
#     filled_data = data.copy()
#     filled_data['S_n'] = filled_data.apply(lambda row: ordering_dic[row['biomarker']], axis = 1)
#     ln_likelihood = 0 
#     for p in range(num_participants):
#         pdata = filled_data[filled_data.participant == p].reset_index(drop=True)
#         average_likelihood = average_all_likelihood(pdata, num_biomarkers, theta_phi)
#         p_ln_likelihood = (
#             # natural logarithm
#            np.log(average_likelihood) 
#            if average_likelihood > 0
#            # this is to avoid np.log(0)
#            else np.log(average_likelihood + 1e-20)
#         )
#         ln_likelihood += p_ln_likelihood
#     return ln_likelihood

def calculate_soft_kmeans_for_biomarker(
        data,
        biomarker,
        order_dict,
        n_participants,
        non_diseased_participants,
        hashmap_of_normalized_stage_likelihood_dicts,
        diseased_stages,
        seed=None
):
    """
    Process soft K-means clustering for a single biomarker.
    
    Parameters:
        data (pd.DataFrame): The data containing measurements.
        biomarker (str): The biomarker to process.
        order_dict (dict): Dictionary mapping biomarkers to their order.
        n_participants (int): Number of participants in the study.
        non_diseased_participants (list): List of non-diseased participants.
        hashmap_of_normalized_stage_likelihood_dicts (dict): Hash map of dictionaries containing stage likelihoods for each participant.
        diseased_stages (list): List of diseased stages.
        seed (int, optional): Random seed for reproducibility.
    
    Returns:
        tuple: Means and standard deviations for affected and non-affected clusters.
    """
    if seed is not None:
        np.random.seed(seed)  # Set random seed for reproducibility

    # DataFrame for this biomarker
    biomarker_df = data[data['biomarker'] == biomarker].reset_index(drop=True)
    # Extract measurements
    measurements = np.array(biomarker_df['measurement'])

    this_biomarker_order = order_dict[biomarker]

    affected_cluster = []
    non_affected_cluster = []

    for p in range(n_participants):
        if p in non_diseased_participants:
            non_affected_cluster.append(measurements[p])
        else:
            if this_biomarker_order == 1:
                affected_cluster.append(measurements[p])
            else:
                normalized_stage_likelihood_dict = hashmap_of_normalized_stage_likelihood_dicts[p]
                # Calculate probabilities for affected and non-affected states
                affected_prob = sum(
                    normalized_stage_likelihood_dict[s] for s in diseased_stages if s >= this_biomarker_order
                )
                non_affected_prob = sum(
                    normalized_stage_likelihood_dict[s] for s in diseased_stages if s < this_biomarker_order
                )
                if affected_prob > non_affected_prob:
                    affected_cluster.append(measurements[p])
                elif affected_prob < non_affected_prob:
                    non_affected_cluster.append(measurements[p])
                else:
                    # Assign to either cluster randomly if probabilities are equal
                    if np.random.rand() > 0.5:
                        affected_cluster.append(measurements[p])
                    else:
                        non_affected_cluster.append(measurements[p])

    # Compute means and standard deviations
    theta_mean = np.mean(affected_cluster) if affected_cluster else np.nan
    theta_std = np.std(affected_cluster) if affected_cluster else np.nan
    phi_mean = np.mean(non_affected_cluster) if non_affected_cluster else np.nan
    phi_std = np.std(non_affected_cluster) if non_affected_cluster else np.nan

    return theta_mean, theta_std, phi_mean, phi_std

def soft_kmeans_theta_phi_estimates(
        iteration,
        prior_theta_phi_estimates,
        data_we_have, 
        biomarkers, 
        order_dict, 
        n_participants, 
        non_diseased_participants, 
        hashmap_of_normalized_stage_likelihood_dicts, 
        diseased_stages, 
        seed=None):
    """
    Get the DataFrame of theta and phi using the soft K-means algorithm for all biomarkers.
    
    Parameters:
        data_we_have (pd.DataFrame): DataFrame containing the data.
        biomarkers (list): List of biomarkers in string.
        order_dict (dict): Dictionary mapping biomarkers to their order.
        n_participants (int): Number of participants in the study.
        non_diseased_participants (list): List of non-diseased participants.
        hashmap_of_normalized_stage_likelihood_dicts (dict): Hash map of dictionaries containing stage likelihoods for each participant.
        diseased_stages (list): List of diseased stages.
        seed (int, optional): Random seed for reproducibility.
    
    Returns:
        pd.DataFrame: DataFrame containing the means and standard deviations for theta and phi for each biomarker.
    """
    # List to store the estimates
    means_stds_estimate_dict_list = []
    for biomarker in biomarkers:
        dic = {'biomarker': biomarker}
        prior_theta_phi_estimates_biomarker = prior_theta_phi_estimates[
            prior_theta_phi_estimates.biomarker == biomarker].reset_index(drop=True)
        theta_mean, theta_std, phi_mean, phi_std = calculate_soft_kmeans_for_biomarker(
            data_we_have, 
            biomarker, 
            order_dict, 
            n_participants, 
            non_diseased_participants, 
            hashmap_of_normalized_stage_likelihood_dicts, 
            diseased_stages, 
            seed
        )
        if theta_std == 0 or math.isnan(theta_std):
            theta_mean = prior_theta_phi_estimates_biomarker.iloc[0, :]['theta_mean']
            theta_std = prior_theta_phi_estimates_biomarker.iloc[0, :]['theta_std']
        if phi_std == 0 or math.isnan(phi_std):
            phi_mean = prior_theta_phi_estimates_biomarker.iloc[0, :]['phi_mean']
            phi_std = prior_theta_phi_estimates_biomarker.iloc[0, :]['phi_std']
        # if theta_mean == 0 or theta_std == 0 or phi_mean == 0 or phi_std == 0:
        #     print(f"{iteration}, {biomarker}, at least one of the variables is zero.")
        # if math.isnan(theta_mean) or math.isnan(theta_std) or math.isnan(phi_mean) or math.isnan(phi_std):
        #     print(f"{iteration}, {biomarker}, at least one of the variables is nan.")
        dic['theta_mean'] = theta_mean
        dic['theta_std'] = theta_std
        dic['phi_mean'] = phi_mean
        dic['phi_std'] = phi_std
        means_stds_estimate_dict_list.append(dic)
    # print("theta_phi_estimates updated")
    return pd.DataFrame(means_stds_estimate_dict_list)

def calculate_all_participant_ln_likelihood_and_update_hashmap(
        data_we_have,
        current_order_dict,
        n_participants,
        non_diseased_participant_ids,
        theta_phi_estimates,
        diseased_stages,
):
    data = data_we_have.copy()
    data['S_n'] = data.apply(lambda row: current_order_dict[row['biomarker']], axis = 1)
    all_participant_ln_likelihood = 0 
    hashmap_of_normalized_stage_likelihood_dicts = {}
    for p in range(n_participants):
        pdata = data[data.participant == p].reset_index(drop=True)
        if p in non_diseased_participant_ids:
            this_participant_likelihood = compute_likelihood(
            pdata, k_j=0, theta_phi = theta_phi_estimates)
            this_participant_ln_likelihood = np.log(this_participant_likelihood + 1e-10)
        else:
            # initiaze stage_likelihood
            stage_likelihood_dict = dict(zip(diseased_stages, [0]*len(diseased_stages)))
            for k_j in diseased_stages:
                kj_likelihood = compute_likelihood(pdata, k_j, theta_phi_estimates)
                # update each stage likelihood for this participant
                stage_likelihood_dict[k_j] = kj_likelihood
            likelihood_sum = sum(stage_likelihood_dict.values())
            normalized_stage_likelihood = [l/likelihood_sum for l in stage_likelihood_dict.values()]
            normalized_stage_likelihood_dict = dict(
                zip(diseased_stages, normalized_stage_likelihood))
            hashmap_of_normalized_stage_likelihood_dicts[p] = normalized_stage_likelihood_dict
            this_participant_likelihood = weighted_average_likelihood(
                pdata, diseased_stages, normalized_stage_likelihood_dict, theta_phi_estimates)
            this_participant_ln_likelihood = np.log(this_participant_likelihood + 1e-10)
        all_participant_ln_likelihood += this_participant_ln_likelihood
    return all_participant_ln_likelihood, hashmap_of_normalized_stage_likelihood_dicts

def metropolis_hastings_soft_kmeans(
        data_we_have, 
        iterations, 
        log_folder_name
    ):
    '''Implement the metropolis-hastings algorithm
    Inputs: 
        - data: data_we_have
        - iterations: number of iterations

    Outputs:
        - best_order: a numpy array
        - best_likelihood: a scalar 
    '''
    n_participants = len(data_we_have.participant.unique())
    biomarkers = data_we_have.biomarker.unique()
    n_biomarkers = len(biomarkers)
    n_stages = n_biomarkers + 1
    non_diseased_participant_ids = data_we_have.loc[
        data_we_have.diseased == False].participant.unique()
    diseased_stages = np.arange(start = 1, stop = n_stages, step = 1)
    # obtain the iniial theta and phi estimates
    prior_theta_phi_estimates = get_theta_phi_estimates(
        data_we_have, 
        biomarkers, 
        n_clusters = 2,
        method = "kmeans_only"
    )
    theta_phi_estimates = prior_theta_phi_estimates.copy()

    # initialize empty lists
    all_order_dicts = []
    all_current_accepted_likelihoods = []
    acceptance_count = 0
    all_current_acceptance_ratios = []
    all_current_accepted_order_dicts = []
    terminal_output_strings = []

    current_accepted_order = np.random.permutation(np.arange(1, n_stages))
    current_accepted_order_dict = dict(zip(biomarkers, current_accepted_order))
    current_accepted_likelihood = -np.inf 

    for _ in range(iterations):
        new_order = current_accepted_order.copy()
        # random.shuffle(new_order)
        shuffle_order(new_order, n_shuffle=3)
        current_order_dict = dict(zip(biomarkers, new_order))
        all_participant_ln_likelihood, \
        hashmap_of_normalized_stage_likelihood_dicts = calculate_all_participant_ln_likelihood_and_update_hashmap(
            data_we_have,
            current_order_dict,
            n_participants,
            non_diseased_participant_ids,
            theta_phi_estimates,
            diseased_stages,
        )

        # Now, update theta_phi_estimates using soft kmeans
        theta_phi_estimates = soft_kmeans_theta_phi_estimates(
            _,
            prior_theta_phi_estimates,
            data_we_have, 
            biomarkers, 
            current_order_dict, 
            n_participants, 
            non_diseased_participant_ids, 
            hashmap_of_normalized_stage_likelihood_dicts, 
            diseased_stages, 
            seed=1234
        )

        prob_of_accepting_new_order = np.exp(
            all_participant_ln_likelihood - current_accepted_likelihood)
        
        # it will definitly update at the first iteration
        if np.random.rand() < prob_of_accepting_new_order:
            acceptance_count += 1
            current_accepted_order = new_order
            current_accepted_likelihood = all_participant_ln_likelihood
            current_accepted_order_dict = current_order_dict
 
        all_current_accepted_likelihoods.append(current_accepted_likelihood)
        acceptance_ratio = acceptance_count*100/(_+1)
        all_current_acceptance_ratios.append(acceptance_ratio)
        all_order_dicts.append(current_order_dict)
        all_current_accepted_order_dicts.append(current_accepted_order_dict)

        if (_+1) % 10 == 0:
            formatted_string = (
                f"iteration {_ + 1} done, "
                f"current accepted likelihood: {current_accepted_likelihood}, "
                f"current acceptance ratio is {acceptance_ratio:.2f} %, "
                f"current accepted order is {current_accepted_order_dict}, "
            )
            terminal_output_strings.append(formatted_string)
            print(formatted_string)

    final_acceptance_ratio = acceptance_count/iterations

    terminal_output_filename = f"{log_folder_name}/terminal_output.txt"
    with open(terminal_output_filename, 'w') as file:
        for result in terminal_output_strings:
            file.write(result + '\n')

    save_all_dicts(all_order_dicts, log_folder_name, "all_order")
    save_all_dicts(
        all_current_accepted_order_dicts, 
        log_folder_name, 
        "all_current_accepted_order_dicts")
    save_all_current_accepted(
        all_current_accepted_likelihoods, 
        "all_current_accepted_likelihoods", 
        log_folder_name)
    save_all_current_accepted(
        all_current_acceptance_ratios, 
        "all_current_acceptance_ratios", 
        log_folder_name)
    print("done!")
    return (
        current_accepted_order_dict, 
        all_order_dicts, 
        all_current_accepted_order_dicts, 
        all_current_accepted_likelihoods, 
        all_current_acceptance_ratios, 
        final_acceptance_ratio
    )

"""The following has the max method
"""
# def metropolis_hastings_kmeans(
#         data_we_have, 
#         iterations, 
#         theta_phi_kmeans, 
#         log_folder_name
#     ):
#     '''Implement the metropolis-hastings algorithm
#     Inputs: 
#         - data: data_we_have
#         - iterations: number of iterations

#     Outputs:
#         - best_order: a numpy array
#         - best_likelihood: a scalar 
#     '''
#     n_participants = len(data_we_have.participant.unique())
#     biomarkers = data_we_have.biomarker.unique()
#     n_biomarkers = len(biomarkers)
#     n_stages = n_biomarkers + 1
#     non_diseased_participant_ids = data_we_have.loc[
#         data_we_have.diseased == False].participant.unique()
#     diseased_stages = np.arange(start = 1, stop = n_stages, step = 1)

#     all_order_dicts = []
#     all_current_accepted_likelihoods = []
#     acceptance_count = 0
#     all_current_acceptance_ratios = []
#     all_current_accepted_order_dicts = []
#     terminal_output_strings = []

#     current_accepted_order = np.random.permutation(np.arange(1, n_stages))
#     current_accepted_order_dict = dict(zip(biomarkers, current_accepted_order))
#     current_accepted_likelihood = -np.inf 

#     max_likelihood = - np.inf
#     max_dict = {'max_ll': max_likelihood}

#     for _ in range(iterations):
#         should_revert_to_max_likelihood_order = max_dict['max_ll'] > current_accepted_likelihood
#         if np.random.rand() >= 0.9 and should_revert_to_max_likelihood_order:
#             # shallow copy first.
#             new_order = copy.deepcopy(max_dict["max_ll_order"])
#             print(f"reverting to max_likelihood ({max_dict['max_ll']}) and the associated biomarker order now: {new_order}")
#         else:
#             new_order = current_accepted_order.copy()
#             # random.shuffle(new_order)
#             shuffle_order(new_order, n_shuffle=2)

#         current_order_dict = dict(zip(biomarkers, new_order))

#         all_participant_ln_likelihood = compute_all_participant_ln_likelihood(
#             data_we_have,
#             current_order_dict,
#             n_participants,
#             non_diseased_participant_ids,
#             theta_phi_kmeans,
#             diseased_stages,
#         )
        
#         if all_participant_ln_likelihood > max_dict['max_ll']:
#             max_dict['iteration'] = _+1
#             max_dict["max_ll"] = all_participant_ln_likelihood.copy()
#             max_dict["max_ll_order"] = copy.deepcopy(new_order)

#         prob_of_accepting_new_order = np.exp(
#             all_participant_ln_likelihood - current_accepted_likelihood)
        
#         # it will definitly update at the first iteration
#         if np.random.rand() < prob_of_accepting_new_order:
#             acceptance_count += 1
#             current_accepted_order = new_order
#             current_accepted_likelihood = all_participant_ln_likelihood
#             current_accepted_order_dict = current_order_dict
 
#         all_current_accepted_likelihoods.append(current_accepted_likelihood)
#         acceptance_ratio = acceptance_count*100/(_+1)
#         all_current_acceptance_ratios.append(acceptance_ratio)
#         all_order_dicts.append(current_order_dict)
#         all_current_accepted_order_dicts.append(current_accepted_order_dict)

#         if (_+1) % 10 == 0:
#             formatted_string = (
#                 f"iteration {_ + 1} done, "
#                 f"current accepted likelihood: {current_accepted_likelihood}, "
#                 f"current acceptance ratio is {acceptance_ratio:.2f} %, "
#                 f"current accepted order is {current_accepted_order_dict}, "
#                 f"current max likelihood is {max_dict['max_ll']}"
#             )
#             terminal_output_strings.append(formatted_string)
#             print(formatted_string)

#     final_acceptance_ratio = acceptance_count/iterations

#     terminal_output_filename = f"{log_folder_name}/terminal_output.txt"
#     with open(terminal_output_filename, 'w') as file:
#         for result in terminal_output_strings:
#             file.write(result + '\n')

#     save_all_dicts(all_order_dicts, log_folder_name, "all_order")
#     save_all_dicts(
#         all_current_accepted_order_dicts, 
#         log_folder_name, 
#         "all_current_accepted_order_dicts")
#     save_all_current_accepted(
#         all_current_accepted_likelihoods, 
#         "all_current_accepted_likelihoods", 
#         log_folder_name)
#     save_all_current_accepted(
#         all_current_acceptance_ratios, 
#         "all_current_acceptance_ratios", 
#         log_folder_name)
#     pd.DataFrame([max_dict]).to_csv(f"{log_folder_name}/max_info.csv", index = False)
#     print("done!")
#     return (
#         current_accepted_order_dict, 
#         all_order_dicts, 
#         all_current_accepted_order_dicts, 
#         all_current_accepted_likelihoods, 
#         all_current_acceptance_ratios, 
#         final_acceptance_ratio
#     )

def estimate_params_exact(m0, n0, s0_sq, v0, data):
    '''This is to estimate means and vars based on conjugate priors
    Inputs:
        - data: a vector of measurements 
        - m0: prior estimate of $\mu$.
        - n0: how strongly is the prior belief in $m_0$ is held.
        - s0_sq: prior estimate of $\sigma^2$.
        - v0: prior degress of freedome, influencing the certainty of $s_0^2$.
    
    Outputs:
        - mu estiate, std estimate
    '''
    # Data summary
    sample_mean = np.mean(data)
    sample_size = len(data)
    sample_var = np.var(data, ddof=1)  # ddof=1 for unbiased estimator

    # Update hyperparameters for the Normal-Inverse Gamma posterior
    updated_m0 = (n0 * m0 + sample_size * sample_mean) / (n0 + sample_size)
    updated_n0 = n0 + sample_size
    updated_v0 = v0 + sample_size 
    updated_s0_sq = (1 / updated_v0) * ((sample_size - 1) * sample_var + v0 * s0_sq + 
                    (n0 * sample_size / updated_n0) * (sample_mean - m0)**2)
    updated_alpha = updated_v0/2
    updated_beta = updated_v0*updated_s0_sq/2

    # Posterior estimates
    mu_posterior_mean = updated_m0
    sigma_squared_posterior_mean = updated_beta/updated_alpha

    mu_estimation = mu_posterior_mean
    std_estimation = np.sqrt(sigma_squared_posterior_mean)

    return mu_estimation, std_estimation

def get_theta_phi_conjugate_priors(biomarkers, data_we_have, theta_phi_kmeans):
    '''To get estimated parameters, returns a Pandas DataFrame
    Input:
    - biomarkers: biomarkers 
    - data_we_have: participants data filled with initial or updated participant_stages

    Output: 
    - estimate_means_std_df, just like means_stds_df, containing the estimated mean and std_dev for 
      distribution of biomarker values when the biomarker is affected and not affected

    Note that, there is one bug we need to fix: Sometimes, data_full might have only one observation or no ob
    '''
     # empty list of dictionaries to store the estimates 
    means_stds_estimate_dict_list = []
    
    for biomarker in biomarkers: 
        dic = {'biomarker': biomarker}  # Initialize dictionary outside the inner loop
        for affected in [True, False]:
            data_full = data_we_have[(data_we_have.biomarker == biomarker) & (
            data_we_have.affected == affected)]
            if len(data_full) > 1:
                measurements = data_full.measurement
                s0_sq = np.var(measurements, ddof=1)
                m0 = np.mean(measurements)
                mu_estimate, std_estimate = estimate_params_exact(
                    m0 = m0, n0 = 1, s0_sq = s0_sq, v0 = 1, data=measurements)
                if affected:
                    dic['theta_mean'] = mu_estimate
                    dic['theta_std'] = std_estimate
                else:
                    dic['phi_mean'] = mu_estimate
                    dic['phi_std'] = std_estimate
            # If there is only one observation or not observation at all, resort to theta_phi_kmeans
            # YES, IT IS POSSIBLE THAT DATA_FULL HERE IS NULL
            # For example, if a biomarker indicates stage of (num_biomarkers), but all participants' stages
            # are smaller than that stage; so that for all participants, this biomarker is not affected
            else:
                # DONT FORGTE RESET_INDEX; this because you are acessing [0]
                theta_phi_kmeans_biomarker_row = theta_phi_kmeans[
                    theta_phi_kmeans.biomarker == biomarker].reset_index(drop=True)
                if affected:
                    dic['theta_mean'] = theta_phi_kmeans_biomarker_row['theta_mean'][0]
                    dic['theta_std'] = theta_phi_kmeans_biomarker_row['theta_std'][0]
                else:
                    dic['phi_mean'] = theta_phi_kmeans_biomarker_row['phi_mean'][0]
                    dic['phi_std'] = theta_phi_kmeans_biomarker_row['phi_std'][0]
        # print(f"biomarker {biomarker} done!")
        means_stds_estimate_dict_list.append(dic)
    estimate_means_stds_df = pd.DataFrame(means_stds_estimate_dict_list)
    return estimate_means_stds_df 

def add_kj_and_affected_and_modify_diseased(data, participant_stages, n_participants):
    '''This is to fill up data_we_have. 
    Basically, add two columns: k_j, affected, and modify diseased column
    based on the initial or updated participant_stages
    Note that we assume here we've already got S_n

    Inputs:
        - data_we_have
        - participant_stages: np array 
        - participants: 0-99
    '''
    participant_stage_dic = dict(zip(np.arange(0, n_participants), participant_stages))
    data['k_j'] = data.apply(lambda row: participant_stage_dic[row.participant], axis = 1)
    data['diseased'] = data.apply(lambda row: row.k_j > 0, axis = 1)
    data['affected'] = data.apply(lambda row: row.k_j >= row.S_n, axis = 1)
    return data

# def add_kj_and_affected(data_we_have, participant_stages, num_participants):
#     '''This is to fill up data_we_have. 
#     Basically, add two columns: k_j, affected, and modify diseased column
#     based on the initial or updated participant_stages
#     Note that we assume here we've already got S_n

#     Inputs:
#         - data_we_have
#         - participant_stages: np array 
#         - participants: 0-99
#     '''
#     participant_stage_dic = dict(zip(np.arange(0, num_participants), participant_stages))
#     data_we_have['k_j'] = data_we_have.apply(lambda row: participant_stage_dic[row.participant], axis = 1)
#     # data_we_have['diseased'] = data_we_have.apply(lambda row: row.k_j!= 0, axis = 1)
#     data_we_have['affected'] = data_we_have.apply(lambda row: row.k_j >= row.S_n, axis = 1)
#     return data_we_have 

def shuffle_order(arr, n_shuffle):
    # randomly choose three indices
    indices = random.sample(range(len(arr)), n_shuffle)
    # obtain the elements represented by these three random indices and shuffle these elements
    selected_elements = [arr[i] for i in indices]
    random.shuffle(selected_elements)
    # shuffle the original arr
    for i, index in enumerate(indices):
        arr[index] = selected_elements[i]

def compute_all_participant_ln_likelihood_and_update_participant_stages(
        n_participants,
        data,
        non_diseased_participant_ids,
        estimated_theta_phi,
        diseased_stages,
        participant_stages
):
    all_participant_ln_likelihood = 0 
    for p in range(n_participants):
        # this participant data
        pdata = data[data.participant == p].reset_index(drop=True)

        """If this participant is not diseased (i.e., if we know k_j is equal to 0)
        We still need to compute the likelihood of this participant seeing this sequence of biomarker data
        but we do not need to estimate k_j like below

        We still need to compute the likelihood because we need to add it to all_participant_ln_likelihood
        """
        if p in non_diseased_participant_ids:
            this_participant_likelihood = compute_likelihood(
                pdata, k_j=0, theta_phi = estimated_theta_phi)
            this_participant_ln_likelihood = np.log(this_participant_likelihood + 1e-10)
        else:
            # initiaze stage_likelihood
            stage_likelihood_dict = dict(zip(diseased_stages, [0]*len(diseased_stages)))
            for k_j in diseased_stages:
                # even though data above has everything, it is filled up by random stages
                # we don't like it and want to know the true k_j. All the following is to update participant_stages
                participant_likelihood = compute_likelihood(pdata, k_j, estimated_theta_phi)
                # update each stage likelihood for this participant
                stage_likelihood_dict[k_j] = participant_likelihood
            likelihood_sum = sum(stage_likelihood_dict.values())
            normalized_stage_likelihood = [l/likelihood_sum for l in stage_likelihood_dict.values()]
            normalized_stage_likelihood_dict = dict(zip(diseased_stages, normalized_stage_likelihood))
            # print(normalized_stage_likelihood)
            sampled_stage = np.random.choice(diseased_stages, p = normalized_stage_likelihood)
            participant_stages[p] = sampled_stage   

            # use weighted average likelihood because we didn't know the exact participant stage
            # all above to calculate participant_stage is only for the purpous of calculate theta_phi
            # this_participant_likelihood = average_all_likelihood(pdata, n_biomarkers, estimated_theta_phi)
            this_participant_likelihood = weighted_average_likelihood(
                pdata, diseased_stages, normalized_stage_likelihood_dict, estimated_theta_phi)
            this_participant_ln_likelihood = np.log(this_participant_likelihood + 1e-10)
        """
        All the codes in between are calculating this_participant_ln_likelihood. 
        If we already know kj=0, then
        it's very simple. If kj is unknown, we need to calculate the likelihood of seeing 
        this sequence of biomarker
        data at different stages, and get the relative likelihood before 
        we get a sampled stage (this is for estimating theta and phi). 
        Then we calculate this_participant_ln_likelihood using average likelihood. 
        """
        all_participant_ln_likelihood += this_participant_ln_likelihood
    return all_participant_ln_likelihood

def metropolis_hastings_with_conjugate_priors(
        data_we_have, iterations, log_folder_name):
    n_participants = len(data_we_have.participant.unique())
    biomarkers = data_we_have.biomarker.unique()
    n_biomarkers = len(biomarkers)
    n_stages = n_biomarkers + 1
    diseased_stages = np.arange(start = 1, stop = n_stages, step = 1)
    
    non_diseased_participant_ids = data_we_have.loc[
        data_we_have.diseased == False].participant.unique()
    diseased_participant_ids = data_we_have.loc[data_we_have.diseased == True].participant.unique()

    all_order_dicts = []
    all_current_accepted_likelihoods = []
    acceptance_count = 0
    all_current_acceptance_ratios = []
    all_current_accepted_order_dicts = []
    terminal_output_strings = []
    all_participant_stages_at_the_end_of_each_iteration = []

    # initialize an ordering and likelihood
    # note that it should be a random permutation of numbers 1-10
    current_accepted_order = np.random.permutation(np.arange(1, n_stages))
    current_accepted_order_dict = dict(zip(biomarkers, current_accepted_order))
    current_accepted_likelihood = -np.inf 

    participant_stages = np.zeros(n_participants)
    for idx in range(n_participants):
        if idx in diseased_participant_ids:
            participant_stages[idx] = random.randint(1, len(diseased_stages))

    max_likelihood = - np.inf
    max_dict = {'max_ll': max_likelihood}

    info_dict_list = []

    for _ in range(iterations):
        info_dict = {}
        info_dict['iteration'] = _ + 1
        info_dict['participant_stages_at_the_start_of_this_round'] = participant_stages.copy()
        should_revert_to_max_likelihood_order = max_dict['max_ll'] > current_accepted_likelihood
        info_dict['should_revert_to_max_likelihood_order'] = should_revert_to_max_likelihood_order
        if np.random.rand() >= 0.9 and should_revert_to_max_likelihood_order:
            # shallow copy first.
            new_order = copy.deepcopy(max_dict["max_ll_order"])
            info_dict['actualy_reverted'] = True
            print(f"reverting to max_likelihood ({max_dict['max_ll']}) and the associated biomarker order now: {new_order}")
        else:
            # print(f"should revert: {should_revert_to_max_likelihood_order}")
            # we are going to shuffle new_order below. So it's better to copy first. 
            new_order = current_accepted_order.copy()
            # random.shuffle(new_order)
            shuffle_order(new_order, n_shuffle=3)
        
        current_order_dict = dict(zip(biomarkers, new_order))
        info_dict['new_order_to_test'] = current_order_dict

        # copy the data to avoid modifying the original
        data = data_we_have.copy()
        data['S_n'] = data.apply(lambda row: current_order_dict[row['biomarker']], axis = 1)
        # add kj and affected for the whole dataset based on participant_stages
        # also modify diseased col (because it will be useful for the new theta_phi_kmeans)
        data = add_kj_and_affected_and_modify_diseased(data, participant_stages, n_participants)
        theta_phi_kmeans = get_theta_phi_estimates(data.copy(), biomarkers, n_clusters = 2)
        estimated_theta_phi = get_theta_phi_conjugate_priors(biomarkers, data.copy(), theta_phi_kmeans)

        all_participant_ln_likelihood = compute_all_participant_ln_likelihood_and_update_participant_stages(
            n_participants,
            data,
            non_diseased_participant_ids,
            estimated_theta_phi,
            diseased_stages,
            participant_stages,
        )

        info_dict['new_order_likelihood'] = all_participant_ln_likelihood

        if all_participant_ln_likelihood == max_dict['max_ll']:
            print(f"Same max likelihood found with order: {new_order} and likelihood: {all_participant_ln_likelihood}")

        if all_participant_ln_likelihood > max_dict['max_ll']:
            max_dict['iteration'] = _+1
            max_dict["max_ll"] = all_participant_ln_likelihood.copy()
            max_dict["max_ll_order"] = copy.deepcopy(new_order)

        info_dict['max_likelihood_up_until_now'] = max_dict["max_ll"]

        # ratio = likelihood/best_likelihood
        # because we are using np.log(likelihood) and np.log(best_likelihood)
        # np.exp(a)/np.exp(b) = np.exp(a - b)
        # if a > b, then np.exp(a - b) > 1
        prob_of_accepting_new_order = np.exp(
            all_participant_ln_likelihood - current_accepted_likelihood)
        
        info_dict['all_participant_ln_likelihood_is_larger'] = prob_of_accepting_new_order > 1
        
        # it will definitly update at the first iteration
        if np.random.rand() < prob_of_accepting_new_order:
            acceptance_count += 1
            current_accepted_order = new_order
            current_accepted_likelihood = all_participant_ln_likelihood
            current_accepted_order_dict = current_order_dict

        info_dict['accepted_order'] = current_accepted_order_dict
        info_dict['accepted_likelihood'] = current_accepted_likelihood

        all_participant_stages_at_the_end_of_each_iteration.append(participant_stages.copy())
        all_current_accepted_likelihoods.append(current_accepted_likelihood)
        acceptance_ratio = acceptance_count*100/(_+1)
        all_current_acceptance_ratios.append(acceptance_ratio)
        all_order_dicts.append(current_order_dict)
        all_current_accepted_order_dicts.append(current_accepted_order_dict)

        # if (_+1) % (iterations/10) == 0:
        #     participant_stages_sampled = sampled_row_based_on_column_frequencies(
        #         np.array(all_current_accepted_participant_stages)
        #     )

        # if _ >= burn_in and _ % thining == 0:
        if (_+1) % 10 == 0:
            formatted_string = (
                f"iteration {_ + 1} done, "
                f"current accepted likelihood: {current_accepted_likelihood}, "
                f"current acceptance ratio is {acceptance_ratio:.2f} %, "
                f"current accepted order is {current_accepted_order_dict}, "
                f"current max likelihood is {max_dict['max_ll']}"
            )
            terminal_output_strings.append(formatted_string)
            print(formatted_string)

        info_dict['participant_stages_at_the_end_of_iteration'] = participant_stages.copy()
        info_dict['acceptance_ratio'] = acceptance_ratio
        info_dict_list.append(info_dict)

    final_acceptance_ratio = acceptance_count/iterations

    terminal_output_filename = f"{log_folder_name}/terminal_output.txt"
    with open(terminal_output_filename, 'w') as file:
        for result in terminal_output_strings:
            file.write(result + '\n')

    save_all_dicts(all_order_dicts, log_folder_name, "all_order")
    save_all_dicts(
        all_current_accepted_order_dicts, 
        log_folder_name, 
        "all_current_accepted_order_dicts")
    save_all_current_accepted(
        all_current_accepted_likelihoods, 
        "all_current_accepted_likelihoods", 
        log_folder_name)
    save_all_current_accepted(
        all_current_acceptance_ratios, 
        "all_current_acceptance_ratios", 
        log_folder_name)
    save_all_current_participant_stages(
        all_participant_stages_at_the_end_of_each_iteration, 
        "participant_stages_at_the_end_of_each_iteartion", 
        log_folder_name)
    pd.DataFrame(info_dict_list).to_csv(f"{log_folder_name}/info.csv", index = False)
    pd.DataFrame([max_dict]).to_csv(f"{log_folder_name}/max_info.csv", index = False)
    print("done!")
    return (
        current_accepted_order_dict,
        participant_stages,
        all_order_dicts,
        all_participant_stages_at_the_end_of_each_iteration,
        all_current_accepted_order_dicts,
        all_current_accepted_likelihoods,
        all_current_acceptance_ratios,
        final_acceptance_ratio
    )

def get_biomarker_stage_probability(df, burn_in, thining):
    """filter through all_dicts using burn_in and thining 
    and for each biomarker, get probability of being in each possible stage

    Input:
        - df: all_ordering.csv
        - burn_in
        - thinning
    Output:
        - dff: a pandas dataframe where index is biomarker name, each col is each stage
        and each cell is the probability of that biomarker indicating that stage
    """
    df = df[(df.index > burn_in) & (df.index % thining == 0)]
    # Create an empty list to hold dictionaries
    dict_list = []

    # for each biomarker
    for col in df.columns:
        dic = {"biomarker": col}
        # get the frequency of biomarkers
        # value_counts will generate a Series where index is each cell's value
        # and the value is the frequency of that value
        stage_counts = df[col].value_counts()
        # for each stage
        # not that df.shape[1] should be equal to num_biomarkers
        for i in range(1, df.shape[1] + 1):
            # get stage:prabability
            dic[i] = stage_counts.get(i, 0)/len(df)
        dict_list.append(dic)

    dff = pd.DataFrame(dict_list)
    dff.set_index(dff.columns[0], inplace=True)
    return dff 

def save_heatmap(all_dicts, burn_in, thining, folder_name, file_name, title):
    df = pd.DataFrame(all_dicts)
    biomarker_stage_probability_df = get_biomarker_stage_probability(df, burn_in, thining)
    sns.heatmap(biomarker_stage_probability_df, 
                annot=True, cmap="Greys", linewidths=.5, 
                cbar_kws={'label': 'Probability'},
                fmt=".1f",
                # vmin=0, vmax=1,
                )
    plt.xlabel('Stage')
    plt.ylabel('Biomarker')
    plt.title(title)
    plt.savefig(f"{folder_name}/{file_name}.png")
    # plt.savefig(f'{file_name}.pdf')
    plt.close() 

def sampled_row_based_on_column_frequencies(a):
    """for ndarray, sample one element in each col based on elements' frequencies
    input:
        a: a numpy ndarray 
    output:
        a 1d array 
    """
    sampled_row = []
    for col in range(a.shape[1]):
        col_arr = a[:, col]
        unique_elements, counts = np.unique(col_arr, return_counts=True)
        probs = counts/counts.sum()
        sampled_element = np.random.choice(unique_elements, p=probs)
        sampled_row.append(sampled_element)
    return np.array(sampled_row)

def save_all_dicts(all_dicts, log_folder_name, file_name):
    """save all_dicts into a dataframe
    """
    df = pd.DataFrame(all_dicts)
    df['iteration'] = np.arange(start = 1, stop = len(df) + 1, step = 1)
    df.set_index("iteration", inplace=True)
    df.to_csv(f"{log_folder_name}/{file_name}.csv", index=True)

def save_all_current_accepted(var, var_name, log_folder_name):
    """save all_current_order_dicts, all_current_ikelihoods, 
    and all_current_acceptance_ratios
    """
    x = np.arange(start = 1, stop = len(var) + 1, step = 1)
    df = pd.DataFrame({"iteration": x, var_name: var})
    df = df.set_index('iteration')
    df.to_csv(f"{log_folder_name}/{var_name}.csv", index=True)

def save_all_current_participant_stages(var, var_name, log_folder_name):
    df = pd.DataFrame(var)
    df.index.name = 'iteration'
    df.index = df.index + 1
    df.to_csv(f"{log_folder_name}/{var_name}.csv", index=False)

def save_trace_plot(burn_in, all_current_likelihoods, folder_name, file_name, title):
    current_likelihoods_to_plot = all_current_likelihoods[burn_in:]
    x = np.arange(
        start = burn_in + 1, stop = len(all_current_likelihoods) + 1, step = 1)
    plt.scatter(x, current_likelihoods_to_plot, alpha=0.5)
    plt.xlabel('Iteration #')
    plt.ylabel('Current Likelihood')
    plt.title(title)
    plt.savefig(f'{folder_name}/{file_name}.png')
    plt.close() 
