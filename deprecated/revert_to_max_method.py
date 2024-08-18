"""The following has the max method
"""

def metropolis_hastings_with_conjugate_priors(
        data_we_have, iterations, log_folder_name, n_shuffle):
    n_participants = len(data_we_have.participant.unique())
    biomarkers = data_we_have.biomarker.unique()
    n_biomarkers = len(biomarkers)
    n_stages = n_biomarkers + 1
    diseased_stages = np.arange(start = 1, stop = n_stages, step = 1)

    non_diseased_participant_ids = data_we_have.loc[
        data_we_have.diseased == False].participant.unique()
    # diseased_participant_ids = data_we_have.loc[
    #     data_we_have.diseased == True].participant.unique()

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
        if idx not in non_diseased_participant_ids:
            # 1-len(diseased_stages), inclusive on both ends
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

"""The following has the max method
"""
def metropolis_hastings_kmeans(
        data_we_have, 
        iterations, 
        theta_phi_kmeans, 
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

    all_order_dicts = []
    all_current_accepted_likelihoods = []
    acceptance_count = 0
    all_current_acceptance_ratios = []
    all_current_accepted_order_dicts = []
    terminal_output_strings = []

    current_accepted_order = np.random.permutation(np.arange(1, n_stages))
    current_accepted_order_dict = dict(zip(biomarkers, current_accepted_order))
    current_accepted_likelihood = -np.inf 

    max_likelihood = - np.inf
    max_dict = {'max_ll': max_likelihood}

    for _ in range(iterations):
        should_revert_to_max_likelihood_order = max_dict['max_ll'] > current_accepted_likelihood
        if np.random.rand() >= 0.9 and should_revert_to_max_likelihood_order:
            # shallow copy first.
            new_order = copy.deepcopy(max_dict["max_ll_order"])
            print(f"reverting to max_likelihood ({max_dict['max_ll']}) and the associated biomarker order now: {new_order}")
        else:
            new_order = current_accepted_order.copy()
            # random.shuffle(new_order)
            shuffle_order(new_order, n_shuffle=2)

        current_order_dict = dict(zip(biomarkers, new_order))

        all_participant_ln_likelihood = compute_all_participant_ln_likelihood(
            data_we_have,
            current_order_dict,
            n_participants,
            non_diseased_participant_ids,
            theta_phi_kmeans,
            diseased_stages,
        )
        
        if all_participant_ln_likelihood > max_dict['max_ll']:
            max_dict['iteration'] = _+1
            max_dict["max_ll"] = all_participant_ln_likelihood.copy()
            max_dict["max_ll_order"] = copy.deepcopy(new_order)

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
                f"current max likelihood is {max_dict['max_ll']}"
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
    pd.DataFrame([max_dict]).to_csv(f"{log_folder_name}/max_info.csv", index = False)
    print("done!")
    return (
        current_accepted_order_dict, 
        all_order_dicts, 
        all_current_accepted_order_dicts, 
        all_current_accepted_likelihoods, 
        all_current_acceptance_ratios, 
        final_acceptance_ratio
    )