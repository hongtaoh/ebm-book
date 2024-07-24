def metropolis_hastings_with_conjugate_priors(
        data_we_have, iterations, theta_phi_kmeans, log_folder_name):
    num_participants = len(data_we_have.participant.unique())
    num_biomarkers = len(data_we_have.biomarker.unique())
    n_stages = num_biomarkers + 1
    biomarkers = data_we_have.biomarker.unique()
    non_diseased_participants = data_we_have.loc[
        data_we_have.diseased == False].participant.unique()

    all_dicts = []
    all_current_likelihoods = []
    acceptance_count = 0
    all_current_acceptance_ratios = []
    all_current_order_dicts = []
    terminal_output_strings = []
    all_current_participant_stages = []

    # initialize an ordering and likelihood
    # note that it should be a random permutation of numbers 1-10
    current_order = np.random.permutation(np.arange(1, n_stages))
    biomarker_current_order_dic = dict(zip(biomarkers, current_order))
    current_likelihood = -np.inf 

    # initialize participant_stages 
    # note that high should be num_biomarkers + 1; otherwise, no participants will be in the stage of 10
    participant_stages = np.random.randint(low = 0, high = n_stages, size = num_participants)
    participant_stages[non_diseased_participants] = 0

    max_likelihood = - np.inf
    # max_likelihood_ordering_stages_tuple = ()
    max_dict = {}
    # likelihood_stages_dict = {}

    for _ in range(iterations):
        ro_revert_to_max_likelihood_ordering = max_likelihood > current_likelihood
        if np.random.rand() >= 0.9 and ro_revert_to_max_likelihood_ordering:
            new_order = max_dict[max_likelihood]
            print(f"reverting to max_likelihood ({max_likelihood}) and the associated biomarker ordering now: {new_order}")
        else:
            # when we update best_order below,
            # in each iteration, new_order will also update
            new_order = current_order.copy()
        # participant_stages_copy = participant_stages.copy()

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
                stage_likelihood = np.zeros(n_stages)
                for k_j in range(n_stages):
                    # even though data above has everything, it is filled up by random stages
                    # we don't like it and want to know the true k_j. All the following is to update participant_stages

                    # likelihood for this participant to have this specific sequence of biomarker values
                    participant_likelihood = compute_likelihood(pdata, k_j, estimated_theta_phi)

                    # update each stage likelihood for this participant
                    stage_likelihood[k_j] = participant_likelihood
                likelihood_sum = np.sum(stage_likelihood)
                normalized_stage_likelihood = [l/likelihood_sum for l in stage_likelihood]
                sampled_stage = np.random.choice(
                    np.arange(n_stages), p = normalized_stage_likelihood)
                participant_stages[p] = sampled_stage   

                # if participant is in sampled_stage, what is the likelihood of 
                # seeing this sequence of biomarker data:
                # this_participant_likelihood = stage_likelihood[sampled_stage]

                # this_participant_likelihood = average_all_likelihood(pdata, num_biomarkers, estimated_theta_phi)

                # use weighted average likelihood because we didn't know the exact participant stage
                # all above to calculate participant_stage is only for the purpous of calculate theta_phi
                this_participant_likelihood = weighted_average_likelihood(
                    pdata, n_stages, normalized_stage_likelihood, estimated_theta_phi)
                
                # then, update all_participant_likelihood
                if this_participant_likelihood == 0:
                    this_participant_ln_likelihood = np.log(this_participant_likelihood + 1e20)
                else:
                    this_participant_ln_likelihood = np.log(this_participant_likelihood)
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
        
        if all_participant_ln_likelihood > max_likelihood:
            max_likelihood = all_participant_ln_likelihood
            max_dict["max_likelihood"] = max_likelihood
            max_dict["ordering"] = new_order
            max_dict['participant_stages'] = participant_stages

        # ratio = likelihood/best_likelihood
        # because we are using np.log(likelihood) and np.log(best_likelihood)
        # np.exp(a)/np.exp(b) = np.exp(a - b)
        # if a > b, then np.exp(a - b) > 1
        prob_of_accepting_new_order = np.exp(
            all_participant_ln_likelihood - current_likelihood)
        
        # it will definitly update at the first iteration
        if np.random.rand() < prob_of_accepting_new_order:
            acceptance_count += 1
            current_order = new_order
            current_likelihood = all_participant_ln_likelihood
            biomarker_current_order_dic = ordering_dic
            # participant_stages = participant_stages_copy

            # """likelihood_ordering_dict will always keep all_participant_ln_likelihood if
            # all_participant_ln_likelihood is larger than current_likelihood

            # smaller all_participant_ln_likelihood will proportionally kept but that's okay. 
            # """
            # likelihood_ordering_dict[current_likelihood] = current_order
            # # likelihood_stages_dict[current_likelihood] = participant_stages

        all_current_participant_stages.append(participant_stages)
        all_current_likelihoods.append(current_likelihood)
        current_acceptance_ratio = acceptance_count*100/(_+1)
        all_current_acceptance_ratios.append(current_acceptance_ratio)
        all_dicts.append(ordering_dic)
        all_current_order_dicts.append(biomarker_current_order_dic)

        # if (_+1) % (iterations/10) == 0:
        #     participant_stages_sampled = sampled_row_based_on_column_frequencies(
        #         np.array(all_current_participant_stages)
        #     )

        # if _ >= burn_in and _ % thining == 0:
        if (_+1) % 10 == 0:
            formatted_string = (
                f"iteration {_ + 1} done, "
                f"current likelihood: {current_likelihood}, "
                f"current acceptance ratio is {current_acceptance_ratio:.2f} %, "
                f"current order is {biomarker_current_order_dic}"
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
        all_current_order_dicts, log_folder_name, "all_current_order_dicts")
    save_all_current(
        all_current_likelihoods, "all_current_likelihoods", log_folder_name)
    save_all_current(
        all_current_acceptance_ratios, "all_current_acceptance_ratios", log_folder_name)
    save_all_current_participant_stages(
        all_current_participant_stages, "all_current_participant_stages", log_folder_name)
    print("done!")
    return (
        biomarker_current_order_dic,
        participant_stages,
        all_dicts,
        all_current_participant_stages,
        all_current_order_dicts,
        all_current_likelihoods,
        all_current_acceptance_ratios,
        final_acceptance_ratio
    )