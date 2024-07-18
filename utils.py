import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
from scipy.stats import norm
from scipy.stats import mode
from sklearn.cluster import KMeans
from matplotlib.animation import FuncAnimation
import time

def get_theta_phi_using_kmeans(data, biomarker, kmeans_setup):
    """To get theta and phi parametesr for a single biomarker using Kmeans algorithm 
    Input:
        - data: data we have right now, without access to S_n and kj
        - biomarker: a string of biomarker name
    Output:
        mean and std of theta and phi
    """
    # two empty clusters to strore measurements
    clusters = [[] for _ in range(2)]
    # dataframe for this biomarker
    biomarker_df = data[data['biomarker'] == biomarker].reset_index(drop=True)
    measurements = np.array(biomarker_df['measurement'])
    # reshape to satisfy sklearn requirements
    measurements = measurements.reshape(-1, 1)
    # dataframe for non-diseased participants
    healthy_df = biomarker_df[biomarker_df['diseased'] == False]
    kmeans = kmeans_setup.fit(measurements)
    predictions = kmeans.predict(measurements)
    # to store measurements into their cluster
    for i, prediction in enumerate(predictions):
        clusters[prediction].append(measurements[i][0])
    # which cluster are healthy participants in
    healthy_predictions = kmeans.predict(measurements[healthy_df.index])
    # the mode of the above predictions will be the phi cluster index
    phi_cluster_idx = mode(healthy_predictions, keepdims=False).mode
    theta_cluster_idx = 1 - phi_cluster_idx
    theta_mean, theta_std = np.mean(clusters[theta_cluster_idx]), np.std(clusters[theta_cluster_idx])
    phi_mean, phi_std = np.mean(clusters[phi_cluster_idx]), np.std(clusters[phi_cluster_idx])
    return theta_mean, theta_std, phi_mean, phi_std

def get_theta_phi_kmeans(data_we_have, biomarkers, n_clusters,):
    """get the dataframe of theta and phi using kmeans algorithm, for all biomarkers
    """
    kmeans_setup = KMeans(n_clusters, random_state=0, n_init="auto")
    # empty list of dictionaries to store the estimates 
    means_stds_estimate_dict_list = []
    for biomarker in biomarkers:
        dic = {'biomarker': biomarker}
        theta_mean, theta_std, phi_mean, phi_std = get_theta_phi_using_kmeans(data_we_have, biomarker, kmeans_setup)
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
    This function computes the likelihood of seeing this sequence of biomarker values for a specific participant
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

def average_all_likelihood(pdata, num_biomarkers, theta_phi):
    '''This is to compute https://ebm-book2.vercel.app/distributions.html#unknown-k-j
    '''
    return np.mean([compute_likelihood(pdata=pdata, k_j=x, theta_phi=theta_phi) for x in range(num_biomarkers+1)])

def compute_ln_likelihood_assuming_ordering(ordering_dic, data, num_biomarkers, theta_phi):
    """Compute the (ln version of) the likelihood of seeing all participants' data,
    assuming that we already know the ordering
    Inputs:
        - ordering: an array of ordering for biomarker 0-9
        - data: data_we_have
        - num_participants
        - num_biomarkers 
    Outputs:
        - ln(likelihood)
    """
    num_participants = len(data.participant.unique())
    # fill up S_n column using the ordering dict
    # copy first in order not to change data_we_have
    filled_data = data.copy()
    filled_data['S_n'] = filled_data.apply(lambda row: ordering_dic[row['biomarker']], axis = 1)
    ln_likelihood = 0 
    for p in range(num_participants):
        pdata = filled_data[filled_data.participant == p].reset_index(drop=True)
        average_likelihood = average_all_likelihood(pdata, num_biomarkers, theta_phi)
        p_ln_likelihood = (
            # natural logarithm
           np.log(average_likelihood) 
           if average_likelihood > 0
           # this is to avoid np.log(0)
           else np.log(average_likelihood + 1e-20)
        )
        ln_likelihood += p_ln_likelihood
    return ln_likelihood

def metropolis_hastings_theta_phi_kmeans_and_average_likelihood(data_we_have, iterations, theta_phi_kmeans, log_folder_name):
    '''Implement the metropolis-hastings algorithm
    Inputs: 
        - data: data_we_have
        - iterations: number of iterations

    Outputs:
        - best_order: a numpy array
        - best_likelihood: a scalar 
    '''
    biomarkers = data_we_have.biomarker.unique()
    num_biomarkers = len(biomarkers)
    n_stages = num_biomarkers + 1

    all_dicts = []
    all_current_best_likelihoods = []
    acceptance_count = 0
    all_current_acceptance_ratios = []
    all_current_best_order_dicts = []
    terminal_output_strings = []

    # initialize an ordering and likelihood
    # note that it should be a random permutation of numbers 1-10
    current_best_order = np.random.permutation(np.arange(1, n_stages))
    biomarker_current_best_order_dic = dict(zip(biomarkers, current_best_order))
    current_best_likelihood = -np.inf 

    for _ in range(iterations):
        new_order = current_best_order.copy()
        # randomly select two indices
        a, b = np.random.choice(num_biomarkers, 2, replace=False)
        # swapping the order
        new_order[a], new_order[b] = new_order[b], new_order[a]
        # new_order = np.random.permutation(np.arange(1, n_stages))
        biomarker_new_order_dic = dict(zip(biomarkers, new_order))
        ln_likelihood = compute_ln_likelihood_assuming_ordering(
            biomarker_new_order_dic, data_we_have, num_biomarkers, theta_phi_kmeans)
        # if the new order results in a higher likelihood, then update:
        prob_of_accepting_new_order = np.exp(
            ln_likelihood - current_best_likelihood)
        random_number = np.random.rand()
        if random_number < prob_of_accepting_new_order:
            acceptance_count += 1
            current_best_likelihood = ln_likelihood
            biomarker_current_best_order_dic = biomarker_new_order_dic

        all_current_best_likelihoods.append(current_best_likelihood)
        current_acceptance_ratio = acceptance_count*100/(_+1)
        all_current_acceptance_ratios.append(current_acceptance_ratio)
        all_dicts.append(biomarker_new_order_dic)
        all_current_best_order_dicts.append(biomarker_current_best_order_dic)
        
        # if _ >= burn_in and _ % thining == 0:
        #     all_dicts.append(biomarker_new_order_dic)

        if (_+1) % 10 == 0:
            formatted_string = (
                f"iteration {_ + 1} done, "
                f"current best likelihood: {current_best_likelihood}, "
                f"current acceptance ratio is {current_acceptance_ratio:.2f} %, "
                f"current best order is {biomarker_current_best_order_dic}"
            )
            terminal_output_strings.append(formatted_string)
            print(formatted_string)

    final_acceptance_ratio = acceptance_count/iterations

    terminal_output_filename = f"{log_folder_name}/terminal_output.txt"
    with open(terminal_output_filename, 'w') as file:
        for result in terminal_output_strings:
            file.write(result + '\n')

    save_all_dicts(all_dicts, log_folder_name, "all_ordering")
    save_all_dicts(
        all_current_best_order_dicts, log_folder_name, "all_current_best_order_dicts")
    save_all_current_best(
        all_current_best_likelihoods, "all_current_best_likelihoods", log_folder_name)
    save_all_current_best(
        all_current_acceptance_ratios, "all_current_acceptance_ratios", log_folder_name)
    print("done!")
    return (
        biomarker_current_best_order_dic, 
        all_dicts, 
        all_current_best_order_dicts, 
        all_current_best_likelihoods, 
        all_current_acceptance_ratios, 
        final_acceptance_ratio
    )

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

def add_kj_and_affected(data_we_have, participant_stages, num_participants):
    '''This is to fill up data_we_have. 
    Basically, add two columns: k_j, and affected, based on the initial or updated participant_stages
    Note that we assume here we've already got S_n

    Inputs:
        - data_we_have
        - participant_stages: np array 
        - participants: 0-99
    '''
    participant_stage_dic = dict(zip(np.arange(0, num_participants), participant_stages))
    data_we_have['k_j'] = data_we_have.apply(lambda row: participant_stage_dic[row.participant], axis = 1)
    data_we_have['affected'] = data_we_have.apply(lambda row: row.k_j >= row.S_n, axis = 1)
    return data_we_have 

def metropolis_hastings_with_conjugate_priors(
        data_we_have, iterations, theta_phi_kmeans, log_folder_name):
    num_participants = len(data_we_have.participant.unique())
    num_biomarkers = len(data_we_have.biomarker.unique())
    n_stages = num_biomarkers + 1
    biomarkers = data_we_have.biomarker.unique()
    non_diseased_participants = data_we_have.loc[data_we_have.diseased == False].participant.unique()

    all_dicts = []
    all_current_best_likelihoods = []
    acceptance_count = 0
    all_current_acceptance_ratios = []
    all_current_best_order_dicts = []
    terminal_output_strings = []
    all_current_participant_stages = []

    # initialize an ordering and likelihood
    # note that it should be a random permutation of numbers 1-10
    current_best_order = np.random.permutation(np.arange(1, n_stages))
    biomarker_current_best_order_dic = dict(zip(biomarkers, current_best_order))
    current_best_likelihood = -np.inf 

    # initialize participant_stages 
    # note that high should be num_biomarkers + 1; otherwise, no participants will be in the stage of 10
    participant_stages = np.random.randint(low = 0, high = n_stages, size = num_participants)
    participant_stages[non_diseased_participants] = 0

    for _ in range(iterations):
        # when we update best_order below,
        # in each iteration, new_order will also update
        new_order = current_best_order.copy()
        # randomly select two indices
        a, b = np.random.choice(num_biomarkers, 2, replace=False)
        # swapping the order
        new_order[a], new_order[b] = new_order[b], new_order[a]

        # likelihood of seeing all participants' data 
        # biomarker:order dict
        ordering_dic = dict(zip(biomarkers, new_order))
        # fill up S_n column using the ordering dict
        # copy first in order not to change data_we_have
        data = data_we_have.copy()
        # now data_we_have has S_n column
        data['S_n'] = data.apply(lambda row: ordering_dic[row['biomarker']], axis = 1)

        # add kj and affected for the whole dataset based on the initial randomized participant_stages
        data = add_kj_and_affected(data, participant_stages, num_participants)
        # print(data.head())

        # get estimated_theta_phi
        estimated_theta_phi = get_theta_phi_conjugate_priors(
            biomarkers, data, theta_phi_kmeans=theta_phi_kmeans)

        all_participant_ln_likelihood = 0 
        for p in range(num_participants):
            # this participant data
            pdata = data[data.participant == p].reset_index(drop=True)

            """If this participant is not diseased (i.e., if we know k_j is equal to 0)
            We still need to compute the likelihood of this participant seeing this sequence of biomarker data
            but we do not need to estimate k_j like below

            We still need to compute the likelihood because we need to add it to all_participant_ln_likelihood
            """
            if p in non_diseased_participants:
                this_participant_likelihood = compute_likelihood(
                    pdata, k_j = 0, theta_phi = estimated_theta_phi)
                this_participant_ln_likelihood = np.log(this_participant_likelihood)
            else:
                # initiaze stage_likelihood
                stage_likelihood = np.zeros(num_biomarkers + 1)
                for k_j in range(num_biomarkers +1):
                    # even though data above has everything, it is filled up by random stages
                    # we don't like it and want to know the true k_j. All the following is to update participant_stages

                    # likelihood for this participant to have this specific sequence of biomarker values
                    participant_likelihood = compute_likelihood(pdata, k_j, estimated_theta_phi)

                    # update each stage likelihood for this participant
                    stage_likelihood[k_j] = participant_likelihood
                likelihood_sum = np.sum(stage_likelihood)
                normalized_stage_likelihood = [l/likelihood_sum for l in stage_likelihood]
                sampled_stage = np.random.choice(
                    np.arange(num_biomarkers + 1), p = normalized_stage_likelihood)
                participant_stages[p] = sampled_stage   

                # if participant is in sampled_stage, what is the likelihood of seeing this sequence of biomarker data:
                # this_participant_likelihood = stage_likelihood[sampled_stage]

                # use average likelihood because we didn't know the exact participant stage
                # all above to calculate participant_stage is only for the purpous of calculate theta_phi
                this_participant_likelihood = average_all_likelihood(pdata, num_biomarkers, estimated_theta_phi)

                # then, update all_participant_likelihood
                if this_participant_likelihood == 0:
                    this_participant_ln_likelihood = np.log(this_participant_likelihood + 1e20)
                else:
                    this_participant_ln_likelihood = np.log(this_participant_likelihood)
            """
            All the codes in between are calculating this_participant_ln_likelihood. If we already know kj=0, then
            it's very simple. If kj is unknown, we need to calculate the likelihood of seeing this sequence of biomarker
            data at different stages, and get the relative likelihood before we get a sampled stage (this is for estimating theta and phi). 
            Then we calculate this_participant_ln_likelihood using average likelihood. 
            """
            all_participant_ln_likelihood += this_participant_ln_likelihood
        
        # ratio = likelihood/best_likelihood
        # because we are using np.log(likelihood) and np.log(best_likelihood)
        # np.exp(a)/np.exp(b) = np.exp(a - b)
        # if a > b, then np.exp(a - b) > 1
        prob_of_accepting_new_order = np.exp(
            all_participant_ln_likelihood - current_best_likelihood)
        random_number = np.random.rand()
        # it will definitly update at the first iteration
        if random_number < prob_of_accepting_new_order:
            acceptance_count += 1
            current_best_likelihood = all_participant_ln_likelihood
            biomarker_current_best_order_dic = ordering_dic

        all_current_participant_stages.append(participant_stages)
        all_current_best_likelihoods.append(current_best_likelihood)
        current_acceptance_ratio = acceptance_count*100/(_+1)
        all_current_acceptance_ratios.append(current_acceptance_ratio)
        all_dicts.append(ordering_dic)
        all_current_best_order_dicts.append(biomarker_current_best_order_dic)

        # if (_+1) % (iterations/10) == 0:
        #     participant_stages_sampled = sampled_row_based_on_column_frequencies(
        #         np.array(all_current_participant_stages)
        #     )

        # if _ >= burn_in and _ % thining == 0:
        if (_+1) % 10 == 0:
            formatted_string = (
                f"iteration {_ + 1} done, "
                f"current best likelihood: {current_best_likelihood}, "
                f"current acceptance ratio is {current_acceptance_ratio:.2f} %, "
                f"current best order is {biomarker_current_best_order_dic}"
            )
            terminal_output_strings.append(formatted_string)
            print(formatted_string)

    final_acceptance_ratio = acceptance_count/iterations

    terminal_output_filename = f"{log_folder_name}/terminal_output.txt"
    with open(terminal_output_filename, 'w') as file:
        for result in terminal_output_strings:
            file.write(result + '\n')

    save_all_dicts(all_dicts, log_folder_name, "all_ordering")
    save_all_dicts(
        all_current_best_order_dicts, log_folder_name, "all_current_best_order_dicts")
    save_all_current_best(
        all_current_best_likelihoods, "all_current_best_likelihoods", log_folder_name)
    save_all_current_best(
        all_current_acceptance_ratios, "all_current_acceptance_ratios", log_folder_name)
    save_all_current_participant_stages(
        all_current_participant_stages, "all_current_participant_stages", log_folder_name)
    print("done!")
    return (
        biomarker_current_best_order_dic,
        participant_stages,
        all_dicts,
        all_current_participant_stages,
        all_current_best_order_dicts,
        all_current_best_likelihoods,
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

def save_all_current_best(var, var_name, log_folder_name):
    """save all_current_best_order_dicts, all_current_best_likelihoods, 
    and all_current_acceptance_ratios
    """
    x = np.arange(start = 1, stop = len(var) + 1, step = 1)
    df = pd.DataFrame({"iteration": x, var_name: var})
    df = df.set_index('iteration')
    df.to_csv(f"{log_folder_name}/{var_name}.csv", index=True)

def save_all_current_participant_stages(var, var_name, log_folder_name):
    df = pd.DataFrame(var)
    df.index.name = 'iteration'
    df.to_csv(f"{log_folder_name}/{var_name}.csv", index=True)

def save_trace_plot(all_current_best_likelihoods, folder_name, file_name, title):
    x = np.arange(
        start = 1, stop = len(all_current_best_likelihoods) + 1, step = 1)
    plt.scatter(x, all_current_best_likelihoods, alpha=0.5)
    plt.xlabel('Iteration #')
    plt.ylabel('Current Best Likelihood')
    plt.title(title)
    plt.savefig(f'{folder_name}/{file_name}.png')
    plt.close() 
