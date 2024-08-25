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
from scipy.stats import kendalltau

def plot_acc(tau_dic):
    # # save and draw
    df = pd.DataFrame(list(tau_dic.items()), columns=["Combination", "Tau"])
    df.to_csv("n_and_r/comb_tau.csv", index=False)
    # df = pd.read_csv("n_and_r/comb_tau.csv")
    # Create a scatter plot
    plt.figure(figsize=(8, 6))

    # Create the scatter plot using Seaborn
    sns.scatterplot(
        x="Combination", y="Tau", data=df, s=100, color='skyblue', edgecolor='black')

    plt.ylim(bottom=0)

    # Rotate x-axis text labels for better readability
    # plt.xticks(rotation=60, ha='right')

    # Add labels and title with appropriate font sizes
    plt.xlabel("Combination", fontsize=12)
    plt.ylabel("Kendall's tau", fontsize=12)
    plt.title("Scatter Plot of Combination vs Kendall's tau", fontsize=14)

    # Adjust the layout to prevent overlap
    plt.tight_layout()

    # Add caption and note
    caption = "Figure 1: This scatter plot shows the relationship between different combinations and their Kendall's tau."
    note = "Note: Combination: # of healthy participants / # of participants"

    # Adjust figure layout to make room for the caption and note
    plt.subplots_adjust(bottom=0.25)

    plt.figtext(0.5, 0.1, caption,
                ha="center", fontsize=10, bbox=dict(facecolor='none', edgecolor='none', pad=0))
    plt.figtext(0.5, 0.05, note,
                ha="center", fontsize=9, bbox=dict(facecolor='none', edgecolor='none', pad=0))

    # Show the plot
    plt.savefig("n_and_r/comb_tau.png")

if __name__ == '__main__':
    iterations = 1000
    burn_in = 600
    thining = 20
    n_shuffle = 2
    real_order = [1, 3, 5, 2, 4]
    S_ordering = np.array([
        'HIP-FCI', 'PCC-FCI', 'HIP-GMI', 'FUS-GMI', 'FUS-FCI'
    ])
    real_theta_phi = pd.read_csv('data/means_stds.csv')

    ns = [50, 100, 500, 1000]
    rs = [0.1, 0.25, 0.5]
    uniform_prior = True

    # com_str:tau
    tau_dic = {}
    for n in ns:
        for r in rs:
            comb_str = f"{int(r*n)}|{n}"
            doc_strings = []
            participants_data = utils.generate_data_from_ebm(
                n_participants = n, 
                S_ordering = S_ordering, 
                real_theta_phi = real_theta_phi, 
                healthy_ratio = r,
                seed=1234,
            )
            log_folder_name = f"n_and_r/logs/{comb_str}"
            biomarker_best_order_dic, \
            participant_stages, \
            all_dicts, \
            all_current_participant_stages,\
            all_current_order_dicts, \
            all_current_likelihoods, \
            all_current_acceptance_ratios, \
            final_acceptance_ratio = utils.metropolis_hastings_with_conjugate_priors(
                participants_data, iterations, log_folder_name, n_shuffle, uniform_prior,
            )
            most_likely_order_dic = utils.obtain_most_likely_order(
                all_current_order_dicts, burn_in, thining)
            most_likely_order = list(most_likely_order_dic.values())
            if set(most_likely_order) != set(real_order):
                print(most_likely_order)
                print(real_order)
                raise ValueError("This most likely order has repeated stages or different stages than expected.")
                # doc_strings.append("This most likelihood has repeated stages.")
            tau, p_value = kendalltau(most_likely_order, real_order)
            tau_dic[f"{int(r*n)}/{n}"] = tau
            print(f"{comb_str} completed!")
            print("--------------------------------------------------------------------")
    plot_acc(tau_dic)
    


