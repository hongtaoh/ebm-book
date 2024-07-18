"""Conjugate Priors
    """
    start_time = time.time()
    log_folder_name = "logs/chen_data_conjugate_priors"
    img_folder_name = "img/chen_data_conjugate_priors"
    # conjugate_priors + theta_phi_means (as backup)
    biomarker_best_order_dic_chen_data, \
    participant_stages_chen_data, \
    all_dicts_chen_data, \
    all_current_participant_stages_chen_data,\
    all_current_best_order_dicts_chen_data, \
    all_current_best_likelihoods_chen_data, \
    all_current_acceptance_ratios_chen_data, \
    final_acceptance_ratio_chen_data = utils.metropolis_hastings_with_conjugate_priors(
        data_we_have, iterations, theta_phi_kmeans, log_folder_name
    )
    utils.save_heatmap(
        all_dicts_chen_data, burn_in, thining, 
        folder_name=img_folder_name,
        file_name="heatmap_all_orderings",
        title = "Chen Data with Conjugate Priors, All Orderings"
    )
    utils.save_heatmap(
        all_current_best_order_dicts_chen_data, 
        burn_in=0, thining=1, 
        folder_name=img_folder_name,
        file_name = "heatmap_all_current_best",
        title = "Chen Data with Conjugate Priors, All Current Best Orderings"
    )
    utils.save_trace_plot(
        all_current_best_likelihoods_chen_data, 
        folder_name=img_folder_name,
        file_name="trace_plot",
        title = "Trace Plot, Chen Data with Conjugate Priors"
    )
    end_time = time.time()
    execution_time = (end_time - start_time)/60
    print(f"Execution time: {execution_time} minutes for Chen's data using conjugate priors.")

    """USE CHEN'S DATA
    """
    data_we_have = process_chen_data("data/Chen2016Data.xlsx")
    biomarkers = data_we_have.biomarker.unique()
    num_biomarkers = len(biomarkers)
    num_participant = len(data_we_have.participant.unique())

    theta_phi_kmeans = utils.get_theta_phi_kmeans(data_we_have, biomarkers, n_clusters = 2)

"""Use simulated data below
    """
    original_data = pd.read_csv('data/participant_data.csv')
    original_data['diseased'] = original_data.apply(lambda row: row.k_j > 0, axis = 1)
    data_we_have = original_data.drop(['k_j', 'S_n', 'affected_or_not'], axis = 1)
    theta_phi_kmeans = pd.read_csv("data/estimate_means_stds_kmeans.csv")
    biomarkers = data_we_have.biomarker.unique()
    num_biomarkers = len(biomarkers)

    """theta_phi_means + average_likelihood
    """
    start_time = time.time()
    log_folder_name = "logs/simulated_data_kmeans_only"
    img_folder_name = "img/simulated_data_kmeans_only"
    biomarker_best_order_dic_kmeans_average_likelihood, \
    all_dicts_kmeans_average_likelihood, \
    all_current_best_order_dicts_kmeans_average_likelihood, \
    all_current_best_likelihoods_kmeans_average_likelihood, \
    all_current_acceptance_ratios_kmeans_average_likelihood, \
    final_acceptance_ratio_kmeans_average_likelihood = utils.metropolis_hastings_theta_phi_kmeans_and_average_likelihood(
        data_we_have, iterations, theta_phi_kmeans, log_folder_name
    )

    utils.save_heatmap(
        all_dicts_kmeans_average_likelihood, burn_in, thining,
        folder_name=img_folder_name,
        file_name="heatmap_all_orderings",
        title = "Simulated Data with KMeans Only, All Orderings"
    )
    utils.save_heatmap(
        all_current_best_order_dicts_kmeans_average_likelihood, 
        burn_in=0, thining=1, 
        folder_name=img_folder_name,
        file_name = "heatmap_all_current_best",
        title = "Simulated Data with KMeans Only, All Current Best Orderings"
    )
    utils.save_trace_plot(
        all_current_best_likelihoods_kmeans_average_likelihood, 
        folder_name=img_folder_name, 
        file_name="trace_plot",
        title = "Trace Plot, Simulated Data with KMeans Only"
    )
    end_time = time.time()
    execution_time = (end_time - start_time)/60
    print(f"Execution time: {execution_time} minutes, for using kmeans + average likelihood")

    """conjugate priors
    """
    start_time = time.time()
    log_folder_name = "logs/simulated_data_conjugate_priors"
    img_folder_name = "img/simulated_data_conjugate_priors"
    # conjugate_priors + theta_phi_means (as backup) 
    biomarker_best_order_dic_conjugate_priors, \
    participant_stages_conjugate_priors, \
    all_dicts_conjugate_priors, \
    all_current_participant_stages_conjugate_priors,\
    all_current_best_order_dicts_conjugate_priors, \
    all_current_best_likelihoods_conjugate_priors, \
    all_current_acceptance_ratios_conjugate_priors, \
    final_acceptance_ratio_conjugate_priors = utils.metropolis_hastings_with_conjugate_priors(
        data_we_have, iterations, theta_phi_kmeans, log_folder_name
    )
    utils.save_heatmap(
        all_dicts_conjugate_priors, burn_in, thining,
        folder_name=img_folder_name,
        file_name="heatmap_all_orderings",
        title = "Simulated Data with Conjugate Priors, All Orderings"
    )
    utils.save_heatmap(
        all_current_best_order_dicts_conjugate_priors, 
        burn_in=0, thining=1, 
        folder_name=img_folder_name,
        file_name = "heatmap_all_current_best",
        title = "Simulated Data with Conjugate Priors, All Current Best Orderings"
    )
    utils.save_trace_plot(
        all_current_best_likelihoods_conjugate_priors, 
        folder_name=img_folder_name, 
        file_name="trace_plot",
        title = "Trace Plot, Simulated Data with Conjugate Priors"
    )
    end_time = time.time()
    execution_time = (end_time - start_time)/60
    print(f"Execution time: {execution_time} minutes, for conjugate priors")