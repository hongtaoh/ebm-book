import pandas as pd 
import numpy as np 
import time
import utils 

def get_data_we_have(file):
    """Prepare data for analysis below
    """
    df = pd.read_excel(file)
    df.rename(columns={df.columns[0]: 'participant_category', df.columns[1]: 'participant'}, inplace=True)
    df['diseased'] = df.apply(lambda row: row.participant_category != 'CN', axis = 1)
    df = pd.melt(df, id_vars=['participant_category', "participant", "timestamp", 'diseased'], 
                        value_vars=["FCI(HIP)", "GMI(HIP)", "FCI(Fusi)", "FCI(PCC)", "GMI(FUS)"], 
                        var_name='biomarker', value_name='measurement')
    # convert participant id
    num_participant = len(df.participant.unique())
    participant_string_id_dic = dict(zip(df.participant.unique(), [_ for _ in range(num_participant)]))
    df['participant'] = df.apply(lambda row: participant_string_id_dic[row.participant], axis = 1 )
    return df 

if __name__ == '__main__':
    data_we_have = get_data_we_have("data/Chen2016Data.xlsx")
    biomarkers = list(set(data_we_have.biomarker))
    num_biomarkers = len(biomarkers)
    num_participant = len(data_we_have.participant.unique())

    iterations = 20
    burn_in = 2
    thining = 2

    theta_phi_kmeans = utils.get_theta_phi_kmeans(data_we_have, biomarkers, n_clusters = 2)
    start_time = time.time()
    biomarker_best_order_dic, participant_stages, all_dicts = utils.metropolis_hastings_with_conjugate_priors(
        data_we_have, iterations, burn_in, thining, theta_phi_kmeans
    )
    end_time = time.time()
    execution_time = (end_time - start_time)/60
    print(f"Execution time: {execution_time} minutes")
    utils.save_plot(all_dicts, num_biomarkers, file_name="img/heatmap_biomarker_involvement_in_stages")